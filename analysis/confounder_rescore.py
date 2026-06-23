import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL = ROOT / "evaluations"
IMG = {"Danish": ROOT / "outputs" / "output_danish",
       "English": ROOT / "outputs" / "output_english"}
RES = HERE / "results"; RES.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

DA2EN = {"Frugt": "Fruit", "Biler": "Cars", "Tøj": "Clothes", "Skov": "Forest"}
SUBJECTS = ["Fruit", "Cars", "Clothes", "Forest"]
REFERENCE_EN = {
    "Fruit":   "A close-up photo of a fresh fruit on a table",
    "Cars":    "A car parked on a street",
    "Clothes": "A set of clothes draped over a chair",
    "Forest":  "A landscape photo of a deep forest",
}
MODEL_ID = "openai/clip-vit-base-patch32"
DA_COLOR, EN_COLOR = "#D55E00", "#0072B2"

def filename(lang, idval, subj_raw, eval_type):
    if lang == "Danish":
        return f"{int(idval):03d}_{subj_raw}_{eval_type}_00001_.png"
    n = int(str(idval).replace("ENG_", ""))
    return f"ENG_{n:03d}_{subj_raw}_{eval_type}_00001_.png"

def score_images():
    """Re-score with CLIP. Requires torch + transformers + the cached model."""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} ...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)

    for lang, csv in [("Danish", "evaluated_results_clip_danish.csv"),
                      ("English", "evaluated_results_clip_english.csv")]:
        df = pd.read_csv(EVAL / csv)
        df.columns = [c.replace("﻿", "") for c in df.columns]
        df["clip_own_repro"] = np.nan
        df["clip_vs_enref"] = np.nan
        todo = df[df["Evaluation_Type"].isin(["Baseline_Accuracy", "Robustness"])]
        print(f"\n{lang}: scoring {len(todo)} images ...")
        for i, (idx, row) in enumerate(todo.iterrows(), 1):
            subj = DA2EN.get(row["Subject"], row["Subject"])
            fn = filename(lang, row["ID"], row["Subject"], row["Evaluation_Type"])
            path = IMG[lang] / fn
            if not path.exists():
                print(f"  missing {fn}"); continue
            img = Image.open(path).convert("RGB")
            inputs = proc(text=[str(row["Prompt"]), REFERENCE_EN[subj]],
                          images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
            ie = out.image_embeds
            te = out.text_embeds
            df.at[idx, "clip_own_repro"] = round(F.cosine_similarity(ie, te[0:1]).item(), 4)
            df.at[idx, "clip_vs_enref"]  = round(F.cosine_similarity(ie, te[1:2]).item(), 4)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
        out_csv = EVAL / f"rescored_results_{lang.lower()}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  saved {out_csv.name}")


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + conf) / 2, n - 1)
    return m, m - h, m + h


def paired_gap(en, da):
    en, da = np.asarray(en, float), np.asarray(da, float)
    d = en - da; n = len(d)
    m, lo, hi = mean_ci(d)
    t, p = stats.ttest_rel(en, da)
    return {"n": n, "gap": m, "lo": lo, "hi": hi, "t": t, "p": p,
            "d": d.mean() / d.std(ddof=1)}

def analyse():
    da = pd.read_csv(EVAL / "rescored_results_danish.csv")
    en = pd.read_csv(EVAL / "rescored_results_english.csv")
    for x in (da, en):
        x.columns = [c.replace("﻿", "") for c in x.columns]
    da = da.reset_index(drop=True); en = en.reset_index(drop=True)
    assert (da["Seed"].values == en["Seed"].values).all(), "rows misaligned"

    pair = pd.DataFrame({
        "subject": [DA2EN.get(s, s) for s in da["Subject"]],
        "eval_type": da["Evaluation_Type"].values,
        "own_da": pd.to_numeric(da["clip_own_repro"], errors="coerce").values,
        "own_en": pd.to_numeric(en["clip_own_repro"], errors="coerce").values,
        "ref_da": pd.to_numeric(da["clip_vs_enref"], errors="coerce").values,
        "ref_en": pd.to_numeric(en["clip_vs_enref"], errors="coerce").values,
    })
    base = pair[pair.eval_type == "Baseline_Accuracy"].dropna()

    rows = []
    def add(label, sub):
        go = paired_gap(sub.own_en, sub.own_da)
        gr = paired_gap(sub.ref_en, sub.ref_da)
        reduction = (go["gap"] - gr["gap"]) / go["gap"] * 100 if go["gap"] else np.nan
        rows.append({"scope": label, "n_pairs": go["n"],
                     "gap_own": round(go["gap"], 4), "gap_own_p": go["p"],
                     "gap_ref": round(gr["gap"], 4),
                     "gap_ref_ci": f"[{gr['lo']:.4f}, {gr['hi']:.4f}]",
                     "gap_ref_p": gr["p"], "gap_ref_d": round(gr["d"], 3),
                     "pct_of_gap_from_scorer": round(reduction, 1)})
    add("Baseline — overall", base)
    for s in SUBJECTS:
        add(f"Baseline — {s}", base[base.subject == s])

    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "confound_summary.csv", index=False)
    print("\n=== CONFOUND CONTROL: language gap under two metrics (baseline images) ===")
    show = summ.copy()
    for c in ["gap_own_p", "gap_ref_p"]:
        show[c] = show[c].map(lambda v: f"{v:.1e}")
    print(show.to_string(index=False))
    overall = summ.iloc[0]
    print(f"\nInterpretation: the own-prompt gap is {overall['gap_own']:+.4f}; "
          f"against a common English reference it is {overall['gap_ref']:+.4f}.")
    print(f"≈{overall['pct_of_gap_from_scorer']:.0f}% of the measured gap is "
          f"attributable to the English-centric scorer; the remainder is a "
          f"genuine Danish image-generation deficit.")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["Overall"] + SUBJECTS
    sub = [base] + [base[base.subject == s] for s in SUBJECTS]
    x = np.arange(len(labels)); w = 0.38
    own = [paired_gap(s.own_en, s.own_da) for s in sub]
    ref = [paired_gap(s.ref_en, s.ref_da) for s in sub]
    ax.bar(x - w/2, [g["gap"] for g in own], w, yerr=[g["gap"]-g["lo"] for g in own],
           capsize=4, color="#999999", label="vs own-language prompt (original metric)")
    ax.bar(x + w/2, [g["gap"] for g in ref], w, yerr=[g["gap"]-g["lo"] for g in ref],
           capsize=4, color=EN_COLOR, label="vs common English reference (confound-controlled)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("CLIP gap  (English − Danish)")
    ax.set_title("Language gap before vs after controlling the scorer confound",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_confound_reference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results/confound_summary.csv and figures/fig5_confound_reference.png")

def main():
    analyse_only = "--analyse" in sys.argv or "--analyze" in sys.argv
    rescored = (EVAL / "rescored_results_danish.csv").exists()
    if not analyse_only and not rescored:
        score_images()
    elif not rescored:
        print("No rescored CSVs found — run without --analyse first."); return
    analyse()


if __name__ == "__main__":
    main()
