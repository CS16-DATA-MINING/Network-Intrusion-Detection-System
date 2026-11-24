"""Tests statistiques : EDA (normalité, association à la cible) et comparaison de modèles.

Justification des choix (détaillée dans le rapport) :
* Shapiro-Wilk / D'Agostino K² : vérifier l'hypothèse de normalité avant de
  choisir un test paramétrique (t de Student, ANOVA) ou non paramétrique.
* Mann-Whitney U : comparaison normal vs attaque d'une variable numérique non
  normale (alternative non paramétrique au test t) ; taille d'effet r rang-bisériale.
* Kruskal-Wallis : même question pour les 10 catégories (alternative à l'ANOVA) ;
  taille d'effet epsilon².
* Chi² d'indépendance + V de Cramér : variables catégorielles (proto, state,
  service) vs cible.
* Spearman : corrélations monotones robustes aux queues lourdes.
* Benjamini-Hochberg : contrôle du taux de fausses découvertes (FDR) car des
  dizaines de tests sont réalisés simultanément.
* Friedman + Nemenyi, Wilcoxon signé (Holm), McNemar : comparer les modèles
  sur les mêmes plis / le même jeu de test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """p-values ajustées par la procédure de Benjamini-Hochberg (FDR)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[order])
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def normality_tests(df: pd.DataFrame, cols, n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        x = x.sample(min(n, len(x)), random_state=seed).to_numpy()
        if np.ptp(x) == 0:
            rows.append({"feature": c, "shapiro_W": np.nan, "shapiro_p": np.nan,
                         "dagostino_K2": np.nan, "dagostino_p": np.nan,
                         "skewness": 0.0, "kurtosis": 0.0})
            continue
        w, pw = stats.shapiro(x)
        k2, pk = stats.normaltest(x)
        rows.append({"feature": c, "shapiro_W": w, "shapiro_p": pw, "dagostino_K2": k2,
                     "dagostino_p": pk, "skewness": stats.skew(x), "kurtosis": stats.kurtosis(x)})
    out = pd.DataFrame(rows)
    valid = out["shapiro_p"].notna()
    out.loc[valid, "shapiro_p_bh"] = benjamini_hochberg(out.loc[valid, "shapiro_p"])
    out["normal_at_5pct"] = out["shapiro_p_bh"] > 0.05
    return out


def mann_whitney(df: pd.DataFrame, cols, target: str = "label") -> pd.DataFrame:
    rows = []
    g0, g1 = df[df[target] == 0], df[df[target] == 1]
    for c in cols:
        a = pd.to_numeric(g0[c], errors="coerce").dropna()
        b = pd.to_numeric(g1[c], errors="coerce").dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = 1 - 2 * u / (len(a) * len(b))          # corrélation rang-bisériale
        rows.append({"feature": c, "median_normal": a.median(), "median_attack": b.median(),
                     "U": u, "p_value": p, "rank_biserial_r": r, "abs_effect": abs(r)})
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant"] = out["p_bh"] < 0.05
    out["effect_size"] = pd.cut(out["abs_effect"], [-1, 0.1, 0.3, 0.5, 1.01],
                                labels=["négligeable", "faible", "moyen", "fort"])
    return out.sort_values("abs_effect", ascending=False).reset_index(drop=True)


def kruskal_wallis(df: pd.DataFrame, cols, group: str = "attack_cat") -> pd.DataFrame:
    rows = []
    groups = [g for _, g in df.groupby(group)]
    n = len(df)
    k = len(groups)
    for c in cols:
        samples = [pd.to_numeric(g[c], errors="coerce").dropna().to_numpy() for g in groups]
        if all(np.ptp(s) == 0 for s in samples if len(s)):
            continue
        h, p = stats.kruskal(*samples)
        eps2 = (h - k + 1) / (n - k)
        rows.append({"feature": c, "H": h, "p_value": p, "epsilon_sq": max(eps2, 0.0)})
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant"] = out["p_bh"] < 0.05
    return out.sort_values("epsilon_sq", ascending=False).reset_index(drop=True)


def chi2_tests(df: pd.DataFrame, cols, target: str = "label", min_count: int = 5) -> pd.DataFrame:
    rows = []
    for c in cols:
        # Condition de validité du χ² : effectifs théoriques ≥ 5. Une modalité de total r
        # a un effectif théorique minimal r·(plus petite proportion de classe) ; on regroupe
        # dans « autre » les modalités qui ne l'atteignent pas.
        col_prop = df[target].value_counts(normalize=True).min()
        min_row = int(np.ceil(min_count / col_prop))
        counts = df[c].value_counts()
        keep = counts.index[counts >= min_row]
        x = df[c].where(df[c].isin(keep), "autre")
        ct = pd.crosstab(x, df[target])
        if "autre" in ct.index and ct.loc["autre"].sum() < min_row:
            ct = ct.drop(index="autre")
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        n = ct.to_numpy().sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rows.append({"feature": c, "target": target, "chi2": chi2, "dof": dof, "p_value": p,
                     "cramers_v": v, "n_levels": ct.shape[0],
                     "pct_expected_lt5": float((exp < 5).mean() * 100)})
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out


def spearman_matrix(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")


def spearman_with_target(df: pd.DataFrame, cols, target: str = "label") -> pd.DataFrame:
    rows = []
    for c in cols:
        rho, p = stats.spearmanr(pd.to_numeric(df[c], errors="coerce"), df[target],
                                 nan_policy="omit")
        rows.append({"feature": c, "spearman_rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p_value"].fillna(1).to_numpy())
    return out.sort_values("spearman_rho", key=np.abs, ascending=False).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Diagnostic IQR (Tukey) : proportion de valeurs hors [Q1-1.5IQR, Q3+1.5IQR]."""
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_mask = (x < lo) | (x > hi)
        rows.append({"feature": c, "q1": q1, "q3": q3, "iqr": iqr,
                     "pct_outliers": 100 * out_mask.mean(),
                     "pct_outliers_attack": 100 * out_mask[df.loc[x.index, "label"] == 1].mean()
                     if "label" in df else np.nan})
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)


# ---------------------------------------------------------------------------
# Comparaison de modèles
# ---------------------------------------------------------------------------
_Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
               15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def friedman_nemenyi(fold_scores: pd.DataFrame) -> dict:
    """Test de Friedman sur les scores par pli + distance critique de Nemenyi."""
    fs = fold_scores.dropna(axis=1)
    k, n = fs.shape[1], fs.shape[0]
    stat, p = stats.friedmanchisquare(*[fs[c] for c in fs.columns])
    ranks = fs.rank(axis=1, ascending=False).mean().sort_values()
    q = _Q_ALPHA_05.get(k, 3.544)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"chi2": float(stat), "p_value": float(p), "k_models": int(k), "n_folds": int(n),
            "critical_difference": float(cd), "mean_ranks": ranks.round(3).to_dict()}


def wilcoxon_vs_best(fold_scores: pd.DataFrame) -> pd.DataFrame:
    fs = fold_scores.dropna(axis=1)
    best = fs.mean().idxmax()
    rows = []
    for c in fs.columns:
        if c == best:
            continue
        d = fs[best] - fs[c]
        if np.allclose(d, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(fs[best], fs[c], zero_method="zsplit")
        rows.append({"best": best, "model": c, "mean_diff": float(d.mean()),
                     "W": stat, "p_value": float(p)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"].to_numpy())
        out["significant"] = out["p_holm"] < 0.05
    return out


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Test de McNemar (avec correction de continuité) entre deux classifieurs."""
    a_ok, b_ok = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    if b + c < 25:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return {"b": b, "c": c, "chi2": np.nan, "p_value": float(p), "exact": True}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)),
            "exact": False}
