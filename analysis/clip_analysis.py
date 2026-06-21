from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_DIR = ROOT / "evaluations"
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_MAP = {"Frugt": "Fruit", "Biler": "Cars", "Tøj": "Clothes", "Skov": "Forest"}
SUBJECT_ORDER = ["Fruit", "Cars", "Clothes", "Forest"]

VARIATION_TYPE = {0: "Word order", 1: "Synonym", 2: "Typos", 3: "Phrasing", 4: "Keyword"}

DA_COLOR = "#D55E00"
EN_COLOR = "#0072B2"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.autolayout": True
})

def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan, n
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + conf) / 2, n - 1)
    return m, m - h, m + h, n

def cohen_d_paired(diff):
    diff = np.asarray(diff, float)
    return diff.mean() / diff.std(ddof=1)

def cohen_d_independent(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp

def d_interpretation(d):
    a = abs(d)
    return ("insignificant" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large")

def load_paired():
    da = pd.read_csv(EVAL_DIR / "evaluated_results_danish.csv")
    en = pd.read_csv(EVAL_DIR / "evaluated_results_english.csv")
    da = da.reset_index(drop=True)
    en = en.reset_index(drop=True)

    paired = pd.DataFrame({
        "subject": en["Subject"].values,
        "subject_da": da["Subject"].values,
        "eval_type": en["Evaluation_Type"].values,
        "seed": en["Seed"].values,
        "prompt_da": da["Prompt"].values,
        "prompt_en": en["Prompt"].values,
        "clip_da": pd.to_numeric(da["CLIP_Score"], errors="coerce").values,
        "clip_en": pd.to_numeric(en["CLIP_Score"], errors="coerce").values,
    })

    paired["var_type"] = pd.Series([pd.NA] * len(paired), dtype="object")
    for subj in paired["subject"].unique():
        mask = (paired["subject"] == subj) & (paired["eval_type"] == "Robustness")
        block = paired[mask]
        order = {p: i for i, p in enumerate(pd.unique(block["prompt_en"]))}
        paired.loc[mask, "var_type"] = block["prompt_en"].map(
            lambda p: VARIATION_TYPE[order[p]]
        )
    return paired

def to_long(paired):
    rows = []
    for _, r in paired.iterrows():
        for lang, col in [("Danish", "clip_da"), ("English", "clip_en")]:
            rows.append({"language": lang, "subject": r["subject"],
                         "eval_type": r["eval_type"], "seed": r["seed"],
                         "var_type": r["var_type"], "clip": r[col]})
    return pd.DataFrame(rows)



# Language gap analysis (paired)
def language_gap_analysis(paired):
    out = []

    def run(label, sub):
        sub = sub.dropna(subset=["clip_da", "clip_en"])
        da, en = sub["clip_da"].values, sub["clip_en"].values
        diff = en - da
        m, lo, hi, n = mean_ci(diff)
        t, p_t = stats.ttest_rel(en, da)
        w, p_w = stats.wilcoxon(en, da)
        sh_p = stats.shapiro(diff).pvalue if 3 <= n <= 5000 else np.nan
        d = cohen_d_paired(diff)
        out.append({"comparison": label, "n_pairs": n,
                    "mean_DA": round(da.mean(), 4), "mean_EN": round(en.mean(), 4),
                    "mean_diff": round(m, 4),
                    "95ci_lo": round(lo, 4), "95ci_hi": round(hi, 4),
                    "t": round(t, 3), "p_ttest": p_t,
                    "wilcoxon_p": p_w, "cohen_d": round(d, 3),
                    "effect": d_interpretation(d), "shapiro_p_diff": sh_p})
    
    scored = paired[paired["eval_type"].isin(["Baseline_Accuracy", "Robustness"])]
    run("Overall", scored)
    run("Baseline accuracy", paired[paired["eval_type"] == "Baseline_Accuracy"])
    run("Robustness", paired[paired["eval_type"] == "Robustness"])
    for subj in SUBJECT_ORDER:
        run(f"Subject: {subj}", scored[scored["subject"] == subj])
    
    df = pd.DataFrame(out)
    df.to_csv(RES_DIR / "language_gap_tests.csv", index=False)
    return df


# Accuracy analysis (baseline)
def accuracy_analysis(long):
    base = long[long["eval_type"] == "Baseline_Accuracy"]
    rows = []
    for lang in ["Danish", "English"]:
        for subj in SUBJECT_ORDER:
            x = base[(base.language == lang) & (base.subject == subj)]["clip"]
            m, lo, hi, n = mean_ci(x)
            rows.append({"language": lang, "subject": subj, "n": n,
                         "mean": round(m, 4), "sd": round(np.std(x, ddof=1), 4),
                         "95ci_lo": round(lo, 4), "95ci_hi": round(hi, 4)})
    summ = pd.DataFrame(rows)

    anova_rows = []
    for lang in ["Danish", "English"]:
        groups = [base[(base.language == lang) & (base.subject == s)]["clip"].values for s in SUBJECT_ORDER]
        f, p = stats.f_oneway(*groups)
        kw_h, kw_p = stats.kruskal(*groups)
        anova_rows.append({"language": lang, "F": round(f, 3), "p_anova": p,
                           "kruskal_H": round(kw_h, 3), "p_kruskal": kw_p})
    
    summ.to_csv(RES_DIR / "accuracy_summary.csv", index=False)
    pd.DataFrame(anova_rows).to_csv(RES_DIR / "accuracy_anova.csv", index=False)
    return summ


# Robustness analysis
def robustness_analysis(long):
    rows = []
    for lang in ["Danish", "English"]:
        for subj in SUBJECT_ORDER + ["ALL"]:
            sel = long[long.language == lang]
            if subj != "ALL":
                sel = sel[sel.subject == subj]
            b = sel[sel.eval_type == "Baseline_Accuracy"]["clip"].dropna().values
            r = sel[sel.eval_type == "Robustness"]["clip"].dropna().values
            t, p = stats.ttest_ind(b, r, equal_var=False) # Welch's t-test
            d = cohen_d_independent(b, r)
            rows.append({"language": lang, "subject": subj,
                         "n_base": len(b), "n_robust": len(r),
                         "mean_base": round(b.mean(), 4),
                         "mean_robust": round(r.mean(), 4),
                         "drop_base_minus_robust": round(b.mean() - r.mean(), 4),
                         "welch_t": round(t, 3), "p": p,
                         "cohen_d": round(d, 3), "effect": d_interpretation(d)})
    base_vs_robust = pd.DataFrame(rows)
    base_vs_robust.to_csv(RES_DIR / "robustness_baseline_vs_pooled.csv", index=False)

    # per variation vs baseline
    rows = []
    for lang in ["Danish", "English"]:
        sel = long[long.language == lang]
        base = sel[sel.eval_type == "Baseline_Accuracy"]["clip"].dropna().values
        for vr in VARIATION_TYPE.values():
            x = sel[(sel.eval_type == "Robustness") & (sel.var_type == vr)]["clip"].dropna().values
            m, lo, hi, n = mean_ci(x)
            t, p = stats.ttest_ind(base, x, equal_var=False)
            d = cohen_d_independent(base, x)
            rows.append({"language": lang, "variation": vr, "n": n,
                         "mean": round(m, 4), "95ci_lo": round(lo, 4),
                         "95ci_hi": round(hi, 4),
                         "vs_baseline_drop": round(base.mean() - m, 4),
                         "welch_t": round(t, 3), "p_vs_baseline": p,
                         "cohen_d": round(d, 3), "effect": d_interpretation(d)})
    variations = pd.DataFrame(rows)
    variations.to_csv(RES_DIR / "robustness_by_perturbation.csv", index=False)
    return base_vs_robust, variations