"""LLM-side pipeline stages. All stages are resumable through the prompt cache.

  python run.py audit   TASK [MODEL]   # auditor labels for pool + random records
  python run.py detect  TASK           # on-device detector (Presidio + lexicon)
  python run.py cf      TASK           # counterfactual single-record utilities, training users
  python run.py reader  TASK [MODEL]   # reader outputs for every context listed by gate.py
  python run.py judge   TASK [MODEL]   # LaMP-QA rubric judge over all cached reader outputs
"""
import json
import os
import re
import sys

from common import (AUDIT_MULTI_BATCH_PROMPT, AUDITOR, K, NUM_PREDICT, READER, RES, LLMCache,
                    audit_batch, audit_record, generate, load_proc, reader_prompt, record_text, utility)

M_GATE = 8   # the gate re-ranks the top-8 retrieved records
N_READER = int(os.environ.get("N_READER", 150))
M_CF = 6     # counterfactual labels are collected for the top-6 retrieved records
WORKERS = int(os.environ.get("WORKERS", 4))
AUDIT_BATCH_SIZE = int(os.environ.get("AUDIT_BATCH_SIZE", 5))
AUDIT_CROSS_USER = os.environ.get("AUDIT_CROSS_USER", "0") == "1"


def pmap(fn, items):
    """Ordered map with a small thread pool (the LLM server serialises requests)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(WORKERS) as ex:
        yield from ex.map(fn, items)


def audit_ids(u, with_rand=True):
    ids = set(range(min(M_GATE, u["n_pool"])))
    if with_rand:
        ids |= set(u["rand"])
    return sorted(ids)


def tag(model):
    return model.replace(":", "_").replace(".", "")


def stage_audit(task, model=AUDITOR, sample=None, eval_top=False):
    """One auditor call per record. Random-baseline records are audited for reader users only.
    eval_top=True audits only the top-M_GATE candidates of evaluation users (second auditor)."""
    cache = LLMCache(f"audit_{task}_{tag(model)}")
    users = load_proc(task)["users"]
    n_ev = 0
    todo = []
    for u in users:
        if eval_top:
            if u["split"] == "eval":
                todo += [(u, ci) for ci in audit_ids(u, with_rand=False)]
            continue
        rd = u["split"] == "eval" and n_ev < N_READER
        n_ev += u["split"] == "eval"
        todo += [(u, ci) for ci in audit_ids(u, with_rand=rd)]
    if sample:  # fixed validation sample for a second auditor
        import numpy as np
        rng = np.random.default_rng(7)
        todo = [todo[i] for i in rng.permutation(len(todo))[:sample]]
    out = {}
    # Batch records from the same user to reduce inference overhead.  The batch prompt is
    # still one independent category decision per record; audit_batch falls back to individual
    # calls when the structured response is malformed.  Set AUDIT_BATCH_SIZE=1 to reproduce the
    # historical one-record-per-request schedule.
    if AUDIT_BATCH_SIZE > 1:
        if AUDIT_CROSS_USER:
            # The multi-user prompt explicitly says that records are independent.  This is
            # useful for a slower second auditor and does not alter the per-record taxonomy.
            batches = [todo[i:i + AUDIT_BATCH_SIZE] for i in range(0, len(todo), AUDIT_BATCH_SIZE)]
        else:
            batches = []
            current_uid, current = None, []
            for u, ci in todo:
                if current_uid != u["uid"] or len(current) >= AUDIT_BATCH_SIZE:
                    if current:
                        batches.append(current)
                    current_uid, current = u["uid"], []
                current.append((u, ci))
            if current:
                batches.append(current)

        def do_batch(batch):
            recs = [u["cands"][ci] for _, ci in batch]
            cats = (audit_batch(cache, model, task, recs, prompt_template=AUDIT_MULTI_BATCH_PROMPT)
                    if AUDIT_CROSS_USER else audit_batch(cache, model, task, recs))
            return [((u["uid"], ci), c) for (u, ci), c in zip(batch, cats)]

        for n, rows in enumerate(pmap(do_batch, batches)):
            for (uid, ci), cats in rows:
                out.setdefault(uid, {})[str(ci)] = cats
            if n % 100 == 0:
                print("audit", task, model, "batches", n, len(batches), flush=True)
    else:
        res = pmap(lambda t: audit_record(cache, model, task, t[0]["cands"][t[1]]), todo)
        for n, ((u, ci), cats) in enumerate(zip(todo, res)):
            out.setdefault(u["uid"], {})[str(ci)] = cats
            if n % 500 == 0:
                print("audit", task, model, n, len(todo), flush=True)
    suffix = "" if model == AUDITOR and not sample and not eval_top else f"_{tag(model)}"
    suffix += "_sample" if sample else ("_evaltop" if eval_top else "")
    json.dump(out, open(os.path.join(RES, f"{task}_audit{suffix}.json"), "w"))


LEX = {
    "health": r"\b(diagnos\w*|cancer|diabet\w*|depress\w*|anxiety|surgery|medication|pregnan\w*|asthma|arthritis|allerg\w*|therapy|disabilit\w*|my doctor|chronic|migraine\w*|adhd|autis\w*)\b",
    "finance": r"\b(salary|debt|mortgage|loan|afford|broke|unemploy\w*|laid off|paycheck|rent|bankrupt\w*)\b",
    "family": r"\b(my (husband|wife|son|daughter|kids?|child\w*|mother|mom|father|dad|boyfriend|girlfriend|partner|fianc\w+|ex|sister|brother|boss|manager|coworker|colleague)s?)\b",
    "belief": r"\b(church|christian|muslim|jewish|atheis\w*|republican|democrat|gay|lesbian|bisexual|trans)\b",
    "place": r"\b(i live in|i work (at|for)|my (employer|company|school|university|address|neighbou?rhood))\b",
    "firstperson": r"\b(i|i'm|i've|my|me|we|our)\b",
}


def stage_detect(task):
    """Cheap on-device detector: Presidio entity counts plus a small disclosed lexicon."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    prov = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"}]})
    an = AnalyzerEngine(nlp_engine=prov.create_engine())
    ents = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP", "DATE_TIME", "URL",
            "CREDIT_CARD", "US_SSN", "IP_ADDRESS", "MEDICAL_LICENSE"]
    users = load_proc(task)["users"]
    out = {}
    for n, u in enumerate(users):
        for ci in audit_ids(u):
            t = record_text(task, u["cands"][ci])
            res = [r for r in an.analyze(text=t, language="en", entities=ents) if r.score >= 0.5]
            f = {e: 0 for e in ents}
            for r in res:
                f[r.entity_type] += 1
            for k, pat in LEX.items():
                f["lex_" + k] = len(re.findall(pat, t.lower()))
            f["len"] = len(t.split())
            out.setdefault(u["uid"], {})[str(ci)] = f
        if n % 100 == 0:
            print("detect", task, n, flush=True)
    json.dump(out, open(os.path.join(RES, f"{task}_detect.json"), "w"))


def run_contexts(task, model, items, users_by_id, label):
    """items: list of (uid, tuple(rec_ids)). Returns {uid: {"1,3": output}}."""
    cache = LLMCache(f"reader_{task}_{tag(model)}")
    out = {}
    def work(it):
        uid, ids = it
        return uid, ids, generate(cache, model, reader_prompt(task, users_by_id[uid], ids),
                                  num_predict=NUM_PREDICT[task])

    for n, (uid, ids, o) in enumerate(pmap(work, items)):
        out.setdefault(uid, {})[",".join(map(str, sorted(ids)))] = o
        if n % 500 == 0:
            print(label, task, model, n, len(items), flush=True)
    return out


def stage_cf(task, model=READER):
    users = [u for u in load_proc(task)["users"] if u["split"] == "train"]
    by = {u["uid"]: u for u in users}
    items = []
    for u in users:
        items.append((u["uid"], ()))
        items += [(u["uid"], (ci,)) for ci in range(min(M_CF, u["n_pool"]))]
    out = run_contexts(task, model, items, by, "cf")
    json.dump(out, open(os.path.join(RES, f"{task}_cf_outputs_{tag(model)}.json"), "w"))


def stage_reader(task, model=READER, ctx_file=None):
    users = load_proc(task)["users"]
    by = {u["uid"]: u for u in users}
    ctx = json.load(open(ctx_file or os.path.join(RES, f"{task}_contexts.json")))
    items = [(uid, tuple(ids)) for uid, sets in ctx.items() for ids in sets]
    # interleave by context index so that a partial run still covers users evenly
    out = run_contexts(task, model, items, by, "reader")
    json.dump(out, open(os.path.join(RES, f"{task}_reader_outputs_{tag(model)}.json"), "w"))


def stage_judge(task, model=READER):
    """Scores every cached reader output of LaMP-QA with the rubric judge."""
    assert task == "lampqa"
    users = {u["uid"]: u for u in load_proc(task)["users"]}
    jc = LLMCache(f"judge_{task}")
    n = 0
    for fn in (f"{task}_cf_outputs_{tag(model)}.json", f"{task}_reader_outputs_{tag(model)}.json",
               f"{task}_oracle_outputs_{tag(model)}.json"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        outs = json.load(open(p))
        sc = {}
        flat = [(uid, key, o) for uid, d in outs.items() for key, o in d.items()]
        for uid, key, s in pmap(lambda t: (t[0], t[1], utility(task, users[t[0]], t[2], jc)), flat):
            sc.setdefault(uid, {})[key] = s
            n += 1
            if n % 500 == 0:
                print("judge", n, flush=True)
        json.dump(sc, open(p.replace("_outputs_", "_utility_"), "w"))


def stage_score(task, model=READER):
    """Utilities for the automatic-metric tasks."""
    users = {u["uid"]: u for u in load_proc(task)["users"]}
    for fn in (f"{task}_cf_outputs_{tag(model)}.json", f"{task}_reader_outputs_{tag(model)}.json",
               f"{task}_oracle_outputs_{tag(model)}.json"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        outs = json.load(open(p))
        sc = {uid: {k: utility(task, users[uid], o) for k, o in d.items()} for uid, d in outs.items()}
        json.dump(sc, open(p.replace("_outputs_", "_utility_"), "w"))


if __name__ == "__main__":
    st, task, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if st == "audit":
        if len(rest) > 1 and rest[1] == "evaltop":
            stage_audit(task, rest[0], eval_top=True)
        else:
            stage_audit(task, *(rest[:1] or [AUDITOR]), sample=int(rest[1]) if len(rest) > 1 else None)
    elif st == "detect":
        stage_detect(task)
    elif st == "cf":
        stage_cf(task, *(rest[:1] or [READER]))
    elif st == "reader":
        stage_reader(task, *(rest[:1] or [READER]), ctx_file=rest[1] if len(rest) > 1 else None)
    elif st == "judge":
        stage_judge(task, *(rest[:1] or [READER]))
    elif st == "score":
        stage_score(task, *(rest[:1] or [READER]))
