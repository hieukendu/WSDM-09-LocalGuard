"""LaMP-QA pilot with qwen2.5:14b on 40 users: top-5 minus no profile, with the 7B pilot on the
same users for reference. Decision rule (PLAN.md): run LaMP-QA with 14B only if the gain > 0.03."""
import json
import os

import numpy as np

from common import RES, SEED

p = json.load(open(os.path.join(RES, "lampqa_pilot_utility_qwen25_14b.json")))
u7 = json.load(open(os.path.join(RES, "lampqa_reader_utility_qwen25_7b.json")))
rng = np.random.default_rng(SEED)


def ci(v):
    v = np.asarray(v)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


g14 = [d[k] - d[""] for d in p.values() for k in d if k]
g7 = [u7[uid][k] - u7[uid][""] for uid, d in p.items() for k in d if k]
out = dict(n=len(p), gain14=ci(g14), gain7_same_users=ci(g7), run_full=bool(np.mean(g14) > 0.03))
out["macros"] = {"qaPilotN": len(p), "qaPilotGain": "{:+.3f} [{:+.3f}, {:+.3f}]".format(*out["gain14"]),
                 "qaPilotGainSeven": "{:+.3f} [{:+.3f}, {:+.3f}]".format(*out["gain7_same_users"])}
json.dump(out, open(os.path.join(RES, "qa_pilot.json"), "w"), indent=1)
print(out)
