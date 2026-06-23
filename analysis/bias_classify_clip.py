import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
ROOT = HERE.parent
EVAL = ROOT / "evaluations"
IMG = {"Danish": ROOT / "outputs" / "output_danish",
       "English": ROOT / "outputs" / "output_english"}
CODED = EVAL / "clip_bias.csv"
SUBJECTS = ["Fruit", "Cars", "Clothes", "Forest"]
CAT_VARS = ["category_1", "category_2", "cultural_default"]

DA_TO_EN = {"Frugt": "Fruit", "Biler": "Cars", "Tøj": "Clothes", "Skov": "Forest"}
MODEL_ID = "openai/clip-vit-base-patch32"

CATS = {
    "Fruit": {
        "category_1": {"apple": "a photo of an apple", "banana": "a photo of a banana",
                       "orange": "a photo of an orange", "grapes": "a photo of a bunch of grapes",
                       "strawberry": "a photo of a strawberry", "lemon": "a photo of a lemon",
                       "pear": "a photo of a pear", "peach": "a photo of a peach"},
        "category_2": {"red": "a photo of red fruit", "yellow": "a photo of yellow fruit",
                       "green": "a photo of green fruit", "orange": "a photo of orange-coloured fruit",
                       "purple": "a photo of purple fruit"},
    },
    "Cars": {
        "category_1": {"sedan": "a photo of a sedan car", "SUV": "a photo of an SUV",
                       "sports": "a photo of a sports car", "hatchback": "a photo of a hatchback car",
                       "truck": "a photo of a pickup truck", "van": "a photo of a van",
                       "wagon": "a photo of a station wagon"},
        "category_2": {"modern": "a photo of a modern car", "vintage": "a photo of a vintage classic car"},
    },
    "Clothes": {
        "category_1": {"suit": "a photo of a business suit", "dress": "a photo of a dress",
                       "shirt": "a photo of a shirt", "t-shirt": "a photo of a t-shirt",
                       "trousers": "a photo of trousers", "jacket": "a photo of a jacket or coat",
                       "sweater": "a photo of a sweater"},
        "category_2": {"masculine": "a photo of men's clothing",
                       "feminine": "a photo of women's clothing",
                       "neutral": "a photo of unisex clothing"},
    },
    "Forest": {
        "category_1": {"coniferous": "a photo of a coniferous pine forest",
                       "deciduous": "a photo of a deciduous broadleaf forest",
                       "tropical": "a photo of a tropical jungle",
                       "mixed": "a photo of a mixed forest"},
        "category_2": {"summer": "a green forest in summer",
                       "autumn": "a forest with orange autumn leaves",
                       "winter": "a snowy forest in winter",
                       "spring": "a forest with spring blossom"},
    },
}


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label vectors (categorical)."""
    a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object)
    keep = ~(pd.isna(a) | pd.isna(b))
    a, b = a[keep], b[keep]
    n = len(a)
    if n == 0:
        return np.nan, 0
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum((m[i, :].sum() / n) * (m[:, i].sum() / n) for i in range(k))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return kappa, n


def kappa_label(k):
    if np.isnan(k): return "n/a"
    return ("poor" if k < .20 else "fair" if k < .40 else "moderate" if k < .60
            else "substantial" if k < .80 else "almost perfect")

def gof_uniform(counts):
    """Chi-square goodness-of-fit vs uniform over the observed categories."""
    counts = np.asarray(counts, float)
    k = len(counts)
    if k < 2 or counts.sum() < 5:
        return None
    expected = np.full(k, counts.sum() / k)
    chi2, p = stats.chisquare(counts, expected)
    low = int((expected < 5).sum())
    return {"chi2": chi2, "df": k - 1, "p": p, "n": int(counts.sum()),
            "k": k, "low_expected_cells": low}


def cramers_v(table):
    table = np.asarray(table, float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, c = table.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1))) if n > 0 and min(r, c) > 1 else np.nan

def load_coded(path):
    df = pd.read_csv(path)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    for col in CAT_VARS:
        if col in df.columns:
            df[col] = df[col].replace({"": np.nan, "N-A": np.nan, "N/A": np.nan,
                                       "-": np.nan}).astype(object)
    return df

def filename(lang, idval, subj_raw):
    if lang == "Danish":
        return f"{int(idval):03d}_{subj_raw}_Bias_00001_.png"
    n = int(str(idval).replace("ENG_", ""))
    return f"ENG_{n:03d}_{subj_raw}_Bias_00001_.png"

def clip_scorer():
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    
    def score_fn(pil_image, texts):
        inputs = proc(text=texts, images=pil_image, return_tensors="pt",
                      padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.logits_per_image.squeeze(0).cpu().numpy()
    return score_fn

def analyse(df):

    kap_rows = []
    if set(df["grader"].unique()) >= {1, 2}:
        g1 = df[df.grader == 1].set_index("image_file")
        g2 = df[df.grader == 2].set_index("image_file")
        common = g1.index.intersection(g2.index)
        for var in CAT_VARS + (["acceptable_image"] if "acceptable_image" in df else []):
            if var not in g1: continue
            k, n = cohen_kappa(g1.loc[common, var], g2.loc[common, var])
            kap_rows.append({"variable": var, "n_double_coded": n,
                             "cohen_kappa": round(k, 3), "agreement": kappa_label(k)})
    kappa_df = pd.DataFrame(kap_rows)

    prim = df[df.grader == 1].copy() if (df["grader"] == 1).any() else df.copy()


    gof_rows = []
    for subj in SUBJECTS:
        for lang in ["Danish", "English"]:
            sub = prim[(prim.subject == subj) & (prim.language == lang)]
            for var in CAT_VARS:
                if var not in sub: continue
                counts = sub[var].dropna().value_counts()
                res = gof_uniform(counts.values)
                if res is None: continue
                top = counts.idxmax(); share = counts.max() / counts.sum()
                gof_rows.append({"subject": subj, "language": lang, "variable": var,
                                 "n": res["n"], "categories": res["k"],
                                 "chi2": round(res["chi2"], 2), "df": res["df"],
                                 "p": res["p"], "top_category": top,
                                 "top_share": round(share, 3),
                                 "low_exp_cells": res["low_expected_cells"]})
    gof_df = pd.DataFrame(gof_rows)


    ind_rows = []
    for subj in SUBJECTS:
        for var in CAT_VARS:
            sub = prim[prim.subject == subj]
            if var not in sub: continue
            tab = pd.crosstab(sub[var], sub["language"])
            if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.values.sum() < 10:
                continue
            chi2, p, dofree, exp = stats.chi2_contingency(tab)
            ind_rows.append({"subject": subj, "variable": var,
                             "n": int(tab.values.sum()), "df": dofree,
                             "chi2": round(chi2, 2), "p": p,
                             "cramers_v": round(cramers_v(tab.values), 3),
                             "low_exp_cells": int((exp < 5).sum())})
    ind_df = pd.DataFrame(ind_rows)
    return kappa_df, gof_df, prim, ind_df

def classify(score_fn):
    from PIL import Image
    rows = []
    for lang, csv in [("Danish", "evaluated_results_clip_danish.csv"),
                      ("English", "evaluated_results_clip_english.csv")]:
        df = pd.read_csv(EVAL / csv)
        df.columns = [c.replace("﻿", "") for c in df.columns]
        bias = df[df["Evaluation_Type"] == "Bias"]
        for n, (_, r) in enumerate(bias.iterrows(), 1):
            subj = DA_TO_EN.get(r["Subject"], r["Subject"])
            fn = filename(lang, r["ID"], r["Subject"])
            path = IMG[lang] / fn
            rec = {"image_file": fn, "language": lang, "subject": subj,
                   "bias_prompt": r["Prompt"], "seed": r["Seed"], "grader": 1,
                   "category_1": "", "category_2": "", "cultural_default": "",
                   "acceptable_image": "", "notes": "auto-CLIP-zeroshot"}
            if not path.exists():
                rec["notes"] = "missing-image"; rows.append(rec); continue
            img = Image.open(path).convert("RGB")
            for var in ("category_1", "category_2"):
                labels = list(CATS[subj][var].keys())
                prompts = list(CATS[subj][var].values())
                sims = score_fn(img, prompts)
                rec[var] = labels[int(np.argmax(sims))]
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(CODED, index=False, encoding="utf-8-sig")
    return CODED

def make_figure(prim):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, subj in zip(axes, SUBJECTS):
        sub = prim[prim.subject == subj]
        cats = sub["category_1"].dropna().unique()
        cats = sorted(cats)
        x = np.arange(len(["Danish", "English"]))
        bottom = np.zeros(2)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(cats), 1)))
        for ci, cat in enumerate(cats):
            vals = [((sub.language == l) & (sub.category_1 == cat)).sum()
                    for l in ["Danish", "English"]]
            ax.bar(["Danish", "English"], vals, bottom=bottom, label=str(cat),
                   color=cmap[ci], edgecolor="white")
            bottom += np.array(vals)
        ax.set_title(subj, fontweight="bold"); ax.set_ylabel("images" if subj == "Fruit" else "")
        ax.legend(fontsize=7, ncol=1, frameon=False)
    fig.suptitle("Bias: primary-category distribution by prompt language",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_bias_category_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_stats(path):
    df = load_coded(path)
    kappa_df, gof_df, primary, ind_df = analyse(df)
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    if len(gof_df):
        show = gof_df.copy(); show["p"] = show["p"].map(lambda x: f"{x:.2e}")
        print(show.to_string(index=False))
        gof_df.to_csv(RES / "bias_goodness_of_fit.csv", index=False)
    
    if len(ind_df):
        show = ind_df.copy(); show["p"] = show["p"].map(lambda x: f"{x:.2e}")
        print(show.to_string(index=False))
        ind_df.to_csv(RES / "bias_independence.csv", index=False)
    
    make_figure(primary)

def main():
    analyse_only = "--analyse" in sys.argv or "--analyze" in sys.argv
    if not analyse_only and not CODED.exists():
        classify(clip_scorer())     # needs CLIP; run locally
    run_stats(CODED)

if __name__ == "__main__":
    main()
    
