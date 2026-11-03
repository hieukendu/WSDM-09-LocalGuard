"""Generate supplementary tables solely from aggregate result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["lampqa", "lamp3", "lamp7", "lamp2"]


def main():
    lines = ["# Secondary aggregate results", "", "Exploratory, union-auditor-relative; no human labels. "
             "300 matched user-level splits per task. These are split-averaged held-out means, not confidence bounds.",
             "", "## Uniform and weighted losses", "",
             "| Task | alpha | Loss | Coarse loss / admitted | Fine loss / admitted |",
             "|---|---:|---|---:|---:|"]
    data = {}
    for task in TASKS:
        d = json.loads((ROOT / f"results/{task}_secondary_exposure.json").read_text())
        data[task] = d
        for a in [.2, .5]:
            for loss in ["count", "severity2"]:
                vals = [d["cells"][f"{g}|{loss}|{a}|all|pooled"] for g in ["coarse", "fine"]]
                lines.append(f"| {task} | {a} | {loss} | " + " | ".join(
                    f"{v['mean_test_loss']:.3f} / {v['mean_size']:.3f}" for v in vals) + " |")
    lines += ["", "## Training-fixed bins and finer-grid coverage", "",
              "| Task | Train cut | Short / long users | Within-bin calibration n range | Grid sizes | Missing reader calls |",
              "|---|---:|---:|---:|---:|---|"]
    for task, d in data.items():
        bins = [r for k, r in d["cells"].items() if k.endswith("within_bin") and "|all|" not in k]
        missing = "; ".join(f"{m}: {r['missing_calls']}" for m, r in d["reader_coverage"].items()) or "no completed full reader cell"
        lines.append(f"| {task} | {d['profile_length_cut_training_median']} | {d['bin_counts']['short']} / {d['bin_counts']['long']} | "
                     f"{min(r['n_cal_min'] for r in bins)}–{max(r['n_cal_max'] for r in bins)} | "
                     f"{d['grid_sizes']['coarse']} to {d['grid_sizes']['fine']} | {missing} |")
    lines += ["", "## Cumulative unique disclosures: synthetic arrivals/rankings on real profiles", "",
              "No observed multi-query histories, new retrieval, or reader utility. Fixed top-8 pool only.", "",
              "| Task | alpha | T | Single-prompt calibration: unique records / categories | Cumulative calibration: unique records / categories |",
              "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        d = json.loads((ROOT / f"results/{task}_cumulative_exposure.json").read_text())
        for a in [.2, .5]:
            for t in [1, 2, 5, 10]:
                vals = [d["cells"][f"{a}|{t}|{mode}"] for mode in ["single_prompt_calibration", "cumulative_calibration"]]
                lines.append(f"| {task} | {a} | {t} | " + " | ".join(
                    f"{v['mean_unique_flagged_records']:.3f} / {v['mean_unique_categories']:.3f}" for v in vals) + " |")
    pilot = json.loads((ROOT / "results/lamp2_pilot.json").read_text())
    lines += ["", "## LaMP-2 pilot (not independent confirmation)", "",
              f"n={pilot['n']}; accuracy no-profile={pilot['acc_none']:.3f}, top-5={pilot['acc_top5']:.3f}; "
              f"paired gain={pilot['gain'][0]:+.3f}, 95% bootstrap interval [{pilot['gain'][1]:+.3f}, {pilot['gain'][2]:+.3f}].", ""]
    (ROOT / "SECONDARY-TABLES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
