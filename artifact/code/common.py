"""Shared utilities: paths, cached Ollama client, data loaders, prompts, metrics."""
import hashlib
import json
import os
import re
import threading
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "results", "llm_cache")
for d in (PROC, RES, CACHE):
    os.makedirs(d, exist_ok=True)

OLLAMA = "http://localhost:11434/api/generate"
READER = "qwen2.5:7b"
READER2 = "llama3.1:8b"
AUDITOR = "qwen2.5:7b"
AUDITOR2 = "gemma2:9b"
JUDGE = "qwen2.5:7b"   # same server model as the reader, so that the shared server never swaps models
TASKS = ["lamp3", "lamp7", "lampqa"]
K = 5          # admission cap (records per prompt)
M = 20         # candidate pool retrieved per query
SEED = 20261005


class LLMCache:
    """Append-only JSONL cache keyed by (model, prompt, options)."""

    def __init__(self, name):
        self.path = os.path.join(CACHE, name + ".jsonl")
        self.mem = {}
        self.lock = threading.Lock()
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.mem[r["k"]] = r["v"]
                    except Exception:
                        pass
        self.fh = open(self.path, "a")

    @staticmethod
    def key(model, prompt, opts):
        s = json.dumps([model, prompt, opts], sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def get(self, model, prompt, opts):
        return self.mem.get(self.key(model, prompt, opts))

    def put(self, model, prompt, opts, v, backend="ollama"):
        """Entries written before 2026-10-06 18:30 carry no backend field: for qwen2.5:7b they
        come from Ollama, for gemma2:9b and llama3.1:8b from the llama.cpp server."""
        k = self.key(model, prompt, opts)
        with self.lock:
            self.mem[k] = v
            self.fh.write(json.dumps({"k": k, "v": v, "b": backend}) + "\n")
            self.fh.flush()


def generate(cache, model, prompt, num_predict=8, fmt=None, temperature=0.0, seed=0):
    """Deterministic (temperature 0, fixed seed) generation with caching."""
    opts = {"temperature": temperature, "num_predict": num_predict, "seed": seed}
    ck = dict(opts, fmt=fmt)
    hit = cache.get(model, prompt, ck)
    if hit is not None:
        return hit
    port = os.environ.get("LLAMACPP_PORT_" + re.sub(r"[^A-Za-z0-9]", "_", model))
    if port:  # same GGUF weights served by llama.cpp (the shared Ollama keeps one model resident)
        return _generate_llamacpp(cache, port, model, prompt, opts, ck, fmt)
    body = {"model": model, "prompt": prompt, "stream": False, "options": opts,
            "keep_alive": "30m"}
    if fmt:
        body["format"] = fmt
    for attempt in range(6):
        try:
            r = requests.post(OLLAMA, json=body, timeout=600)
            r.raise_for_status()
            out = r.json()["response"]
            cache.put(model, prompt, ck, out, backend="ollama-0.30.8")
            return out
        except Exception as e:  # server busy / restarted
            time.sleep(5 * (attempt + 1))
            err = e
    raise RuntimeError(f"ollama failed: {err}")


def _generate_llamacpp(cache, port, model, prompt, opts, ck, fmt):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": opts["temperature"],
            "seed": opts["seed"], "max_tokens": opts["num_predict"]}
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            cache.put(model, prompt, ck, out, backend=os.environ.get("LLAMACPP_BACKEND", "llamacpp-0.30.8-bundled"))
            return out
        except Exception as e:
            time.sleep(5 * (attempt + 1))
            err = e
    raise RuntimeError(f"llama.cpp server failed: {err}")


def trunc(text, n):
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


# ---------------------------------------------------------------- record text
REC_WORDS = {"lamp3": 60, "lamp7": 40, "lampqa": 90, "lamp2": 60}


def record_text(task, rec):
    """Text of one personal record as it would appear in a prompt."""
    t = trunc(rec["text"].replace("\n", " "), REC_WORDS[task])
    if task == "lamp3":
        return f"(rating {rec['score']}) {t}"
    if task == "lamp2":
        return f"(tag: {rec['tag']}) {t}"
    return t


# -------------------------------------------------------------------- prompts
def reader_prompt(task, user, rec_ids):
    recs = [user["cands"][i] for i in sorted(rec_ids)]  # canonical order
    if task == "lamp3":
        head = ""
        if recs:
            head = ("Past reviews written by this user with the ratings they gave:\n"
                    + "\n".join(f"- {record_text(task, r)}" for r in recs) + "\n\n")
        return (head + "What rating from 1 to 5 would this user give to the following "
                "review they wrote? Answer with a single digit and nothing else.\n"
                f"Review: {trunc(user['query'], 150)}\nRating:")
    if task == "lamp7":
        head = ""
        if recs:
            head = ("Past tweets written by this user:\n"
                    + "\n".join(f"- {record_text(task, r)}" for r in recs) + "\n\n")
        return (head + "Rewrite the following sentence as a tweet"
                + (" in this user's writing style" if recs else "")
                + ". Output only the tweet.\n"
                f"Sentence: {user['query']}\nTweet:")
    if task == "lampqa":
        # question first and restated last; records are framed as optional background so that
        # the reader neither answers them nor drifts to their topic
        q = f"A user asks: {user['query']}\n\n"
        bg = ""
        if recs:
            bg = ("For context only, here are earlier posts the same user wrote. Use a post only if it "
                  "reveals something about this user's situation that matters for the question; ignore "
                  "posts about other topics. Never answer these posts.\n"
                  + "\n".join(f"- {record_text(task, r)}" for r in recs) + "\n\n")
        return (q + bg + f"Now answer the user's question (\"{trunc(user['query'], 30)}\") in at most "
                "120 words, addressing the aspects this particular user most likely cares about.\nAnswer:")
    if task == "lamp2":
        head = ""
        if recs:
            head = ("Movies this user tagged before, with the tag the user chose:\n"
                    + "\n".join(f"- {record_text(task, r)}" for r in recs) + "\n\n")
        return (head + "Which tag would this user give to the following movie? Choose exactly one of: "
                + ", ".join(LAMP2_TAGS) + ". Answer with the tag only.\n"
                f"Movie: {trunc(user['query'], 150)}\nTag:")
    raise ValueError(task)


LAMP2_TAGS = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy",
              "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary",
              "violence", "true story"]
NUM_PREDICT = {"lamp3": 2, "lamp7": 48, "lampqa": 190, "lamp2": 8}

CATS = ["AGE_GENDER", "RACE_NATIONALITY", "LOCATION", "APPEARANCE", "PARTNER", "FINANCE",
        "OCCUPATION", "FAMILY", "HEALTH", "MENTAL_HEALTH", "EDUCATION", "PET",
        "SEXUAL_ORIENTATION", "NAME_CONTACT"]
# special-category subset used for the stricter "sensitive" exposure measure
SENSITIVE_CATS = {"HEALTH", "MENTAL_HEALTH", "SEXUAL_ORIENTATION", "RACE_NATIONALITY", "FINANCE"}

AUDIT_HEAD = """You are a privacy auditor. You detect SELF-DISCLOSURE: statements in which the author reveals personal information about themselves or about a specific person close to them.

Categories:
AGE_GENDER: the author's age or gender
RACE_NATIONALITY: the author's race, ethnicity or nationality
LOCATION: where the author lives, works, studies or is travelling (city, region, country, address)
APPEARANCE: the author's body, height, weight or looks
PARTNER: the author's wife, husband, girlfriend, boyfriend, or relationship status
FINANCE: the author's income, debts, purchases they can or cannot afford, financial situation
OCCUPATION: the author's job, employer, profession or workplace situation
FAMILY: information about the author's children, parents, siblings or other relatives
HEALTH: physical health conditions, injuries, medication, pregnancy, disability
MENTAL_HEALTH: mental health conditions, therapy, emotional struggles
EDUCATION: the author's school, degree, studies or exams
PET: the author's own pets
SEXUAL_ORIENTATION: the author's sexual orientation
NAME_CONTACT: real names of private persons, phone numbers, emails, private account handles

A text counts only if it reveals a fact about the author's own life or a person close to them. Opinions about products, films, news or public figures, general questions, and generic statements reveal nothing and get an empty list.
"""

AUDIT_PROMPT = AUDIT_HEAD + """
Text: \"\"\"{text}\"\"\"

Reply with JSON only: {{"categories": [category names disclosed in the text, empty list if none]}}"""

AUDIT_BATCH_PROMPT = AUDIT_HEAD + """
Texts written by one user:
{texts}

Judge every text on its own. Reply with JSON only: {{"labels": [{{"id": 1, "categories": [category names disclosed in text 1, empty list if none]}}, ... one entry per text, in order]}}"""


def _cats(x):
    return sorted({c for c in (x or []) if c in CATS})


def audit_record(cache, model, task, rec):
    out = generate(cache, model, AUDIT_PROMPT.format(text=record_text(task, rec)),
                   num_predict=40, fmt="json")
    try:
        return _cats(json.loads(out).get("categories"))
    except Exception:
        return []


def audit_batch(cache, model, task, recs):
    """Audits up to five records of one user in a single call. Returns one category list per record."""
    texts = "\n".join(f'{i+1}. \"\"\"{record_text(task, r)}\"\"\"' for i, r in enumerate(recs))
    out = generate(cache, model, AUDIT_BATCH_PROMPT.format(texts=texts),
                   num_predict=30 * len(recs) + 20, fmt="json")
    try:
        labs = json.loads(out)["labels"]
        if len(labs) == len(recs):
            return [_cats(l.get("categories")) for l in labs]
    except Exception:
        pass
    return [audit_record(cache, model, task, r) for r in recs]  # malformed reply: one call per record


JUDGE_PROMPT = """You grade an answer to a user's question against the aspects that this user needs addressed.

Question: {q}

Answer: {a}

Aspects:
{aspects}

For each aspect give a score: 0 = not addressed, 1 = partially addressed, 2 = fully addressed.
Reply with JSON only: {{"scores": [one integer per aspect, in order]}}"""


def judge_answer(cache, user, answer):
    asp = user["aspects"]
    lines = "\n".join(f"{i+1}. {a['aspect']}: {trunc(a['reason'], 40)}" for i, a in enumerate(asp))
    out = generate(cache, JUDGE, JUDGE_PROMPT.format(q=user["query"], a=answer, aspects=lines),
                   num_predict=40, fmt="json")
    try:
        sc = json.loads(out)["scores"]
        sc = [min(2, max(0, int(x))) for x in sc][:len(asp)]
        if len(sc) < len(asp):
            sc += [0] * (len(asp) - len(sc))
        return sum(sc) / (2.0 * len(asp))
    except Exception:
        return 0.0


# -------------------------------------------------------------------- metrics
_scorer = None


def rouge_l(pred, gold):
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    s = _scorer.score(gold, pred)
    return s["rouge1"].fmeasure, s["rougeL"].fmeasure


def parse_rating(out):
    m = re.search(r"[1-5]", out or "")
    return int(m.group(0)) if m else 3


def utility(task, user, out, judge_cache=None):
    """Per-query utility in [0,1], larger is better."""
    if task == "lamp3":
        return 1.0 - abs(parse_rating(out) - int(user["gold"])) / 4.0
    if task == "lamp7":
        return rouge_l(out.strip().split("\n")[0], user["gold"])[1]
    if task == "lampqa":
        return judge_answer(judge_cache, user, out.strip())
    if task == "lamp2":
        o = (out or "").strip().lower().strip(".").strip()
        return float(o == user["gold"].strip().lower() or o.startswith(user["gold"].strip().lower()))
    raise ValueError(task)


N_TRAIN_USE = int(os.environ.get("N_TRAIN_USE", 120))
N_EVAL_USE = int(os.environ.get("N_EVAL_USE", 300))


def load_proc(task):
    """Users in sampled order; the first N_TRAIN_USE training and N_EVAL_USE evaluation users."""
    with open(os.path.join(PROC, f"{task}.json")) as f:
        d = json.load(f)
    tr = [u for u in d["users"] if u["split"] == "train"][:N_TRAIN_USE]
    ev = [u for u in d["users"] if u["split"] == "eval"][:N_EVAL_USE]
    d["users"] = tr + ev
    return d
