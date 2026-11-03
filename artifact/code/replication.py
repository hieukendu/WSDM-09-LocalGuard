"""Second reader (llama3.1:8b): full analysis on LaMP-3 and the personalization-gain cell on LaMP-QA.
Writes results/replication.json."""
import json
import os

import numpy as np

from common import K, RES, SEED, load_proc

TAG = "llama31_8b"


def main():
    out, mac = {}, {}
    rng = np.random.default_rng(SEED)
    p = os.path.join(RES, f"lampqa_reader_utility_{TAG}.json")
    if os.path.exists(p):
        u2 = json.load(open(p))
        u1 = json.load(open(os.path.join(RES, "lampqa_reader_utility_qwen25_7b.json")))
        rules = json.load(open(os.path.join(RES, "lampqa_rules.json")))
        res = {}
        for name, util in (("qwen", u1), ("llama", u2)):
            ids = [uid for uid in u2]
            key = lambda ids_: ",".join(map(str, sorted(ids_)))
            none = np.array([util[uid][""] for uid in ids])
            for k in (5,):
                v = np.array([util[uid][key(rules[uid]["sets"]["sim"][k])] for uid in ids]) - none
                bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                res[f"{name}_top{k}"] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            res[f"{name}_none"] = float(none.mean())
        out["lampqa"] = res
        for k in (5,):
            v = res[f"llama_top{k}"]
            mac[f"qaLlamaGain{'A' if k == 1 else 'E'}"] = f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
        mac["qaLlamaNone"] = f"{res['llama_none']:.3f}"
    p = os.path.join(RES, f"lamp3_analysis_{TAG}.json")
    if os.path.exists(p):
        a = json.load(open(p))
        x = a["xfit"][a["primary"]]["crc"]["0.2"]
        out["lamp3"] = dict(top5_minus_none=a["fixed"]["top5_minus_none"],
                            xfit02={m: dict(util=v["util"], vs_top5_90=v["vs_top5_90"], kept=v.get("kept"),
                                            sig_loss=v["sig_loss"], size=v["size"]) for m, v in x.items()})
        v = a["fixed"]["top5_minus_none"]
        mac["threeLlamaGain"] = f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
        for m, nm in (("cert_sim", "CSim"), ("sim_thr", "SimThr"), ("sim", "Sim"), ("flt", "Flt")):
            w = x[m]["vs_top5_90"]
            mac[f"threeLlama{nm}VsTop"] = f"{w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]"
        x5 = a["xfit"][a["primary"]]["crc"]["0.5"]
        for m, nm in (("cert_sim", "CSim"), ("sim_thr", "SimThr")):
            w = x5[m]["vs_top5_90"]
            mac[f"threeLlama{nm}VsTopD"] = f"{w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]"
        mac["threeLlamaN"] = a["n_reader"]
        mac["threeLlamaNone"] = f"{a['fixed']['none']['utility'][0]:.3f}"
        for al, L in (("0.2", "B"), ("0.5", "D")):
            q = a["xfit"][a["primary"]]["crc"][al]["cert_sim"]
            if q.get("kept"):
                k = q["kept"]
                mac[f"threeLlamaCSimKept{L}"] = f"{100*k[0]:.0f}\\% [{100*k[1]:.0f}, {100*k[2]:.0f}]"
                mac[f"threeLlamaCSimKeptLow{L}"] = f"{100*q['kept_90'][1]:.0f}\\%"
                mac[f"threeLlamaCSimNonInf{L}"] = "yes" if q.get("noninferior") else "no"
                mac[f"threeLlamaCSimExpo{L}"] = f"{q['expo']:.2f}"
        mac["threeLlamaNSigLossB"] = sum(x[m]["sig_loss"] for m in ("sim", "sim_thr", "flt", "cert_sim"))
    out["macros"] = mac
    json.dump(out, open(os.path.join(RES, "replication.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
