"""Injecte les résultats réels (registry.json, analysis.json, unsupervised.json) dans le README.

Usage : python scripts/update_readme_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


def pct(x, d=2):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{d}f} %"


def f4(x):
    return "—" if x is None or pd.isna(x) else f"{x:.4f}"


def model_table(reg, task, profile):
    rows = pd.DataFrame(reg["metrics"][profile][task])
    best = reg["best"][profile][task]
    if task == "binary":
        rows = rows.sort_values("f1", ascending=False)
        out = ["| Modèle | Famille | F1 CV plis communs ★ | F1 CV recherche | Accuracy | Précision | Rappel | **F1** | ROC-AUC | FPR |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in rows.iterrows():
            name = f"**{r.model}** 🏆" if r.key == best else r.model
            out.append(f"| {name} | {r.family} | {f4(r.get('cv_common_mean'))} | {f4(r.cv_mean)} | {pct(r.accuracy)} | {pct(r.precision)} | "
                       f"{pct(r.recall)} | **{pct(r.f1)}** | {f4(r.roc_auc)} | {pct(r.false_alarm_rate)} |")
    else:
        rows = rows.sort_values("f1_macro", ascending=False)
        out = ["| Modèle | F1-macro plis communs ★ | F1-macro CV recherche | Accuracy | Acc. équilibrée | **F1-macro** | F1 pondéré | ROC-AUC OvR |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in rows.iterrows():
            name = f"**{r.model}** 🏆" if r.key == best else r.model
            out.append(f"| {name} | {f4(r.get('cv_common_mean'))} | {f4(r.cv_mean)} | {pct(r.accuracy)} | {pct(r.balanced_accuracy)} | **{pct(r.f1_macro)}** | "
                       f"{pct(r.f1_weighted)} | {f4(r.roc_auc_ovr)} |")
    return "\n".join(out)


def main():
    reg = json.loads((C.REGISTRY_FILE).read_text(encoding="utf-8"))
    an_p = C.METRICS_DIR / "analysis.json"
    un_p = C.METRICS_DIR / "unsupervised.json"
    an = json.loads(an_p.read_text(encoding="utf-8")) if an_p.exists() else {}
    un = json.loads(un_p.read_text(encoding="utf-8")) if un_p.exists() else {}
    parts = ["### 4.1 Détection binaire (normal / attaque) — jeu de test (18 258 flux), profil complet",
             model_table(reg, "binary", "full"),
             "\n🏆 = modèle déployé. ★ Critère de sélection : moyenne sur **5 plis stratifiés communs à tous les modèles** (y compris vote et stacking) ; le jeu de test n'est jamais utilisé pour choisir.",
             "\n### 4.2 Classification multi-classe (10 catégories) — profil complet",
             model_table(reg, "multiclass", "full"),
             "\n### 4.3 Profil portable (sans `sttl`, `dttl`, `ct_state_ttl`)",
             model_table(reg, "binary", "portable")]
    ms = an.get("model_stats", {})
    if ms:
        b = ms["binary"]
        parts.append(f"\n### 4.4 Comparaison statistique des modèles\n"
                     f"- **Friedman** (binaire, {b['friedman']['k_models']} modèles × 5 plis communs) : χ² = {b['friedman']['chi2']:.2f}, "
                     f"p = {b['friedman']['p_value']:.2e} ; distance critique de Nemenyi = {b['friedman']['critical_difference']:.2f} rang.\n"
                     f"- **McNemar** ({b['mcnemar']['model_a']} vs {b['mcnemar']['model_b']}, test) : b = {b['mcnemar']['b']}, "
                     f"c = {b['mcnemar']['c']}, p = {b['mcnemar']['p_value']:.3g}.")
    if an.get("feature_importance"):
        top = [r["feature"] for r in an["feature_importance"][:8]]
        parts.append("\n### 4.5 Variables les plus importantes (consensus de 5 méthodes : Gini, gain, SHAP, permutation, information mutuelle)\n"
                     + ", ".join(f"`{t}`" for t in top))
    if an.get("eda"):
        e = an["eda"]
        parts.append(f"\n### 4.6 Tests statistiques\n- Normalité rejetée (Shapiro-Wilk, BH) pour **{e['n_normal_rejected']} / {e['n_tests_normality']}** variables "
                     f"→ tests non paramétriques et `log1p`.\n- Mann-Whitney : **{e['mw_significant']} / {e['mw_total']}** variables significatives "
                     f"(normal vs attaque) ; Kruskal-Wallis : **{e['kw_significant']} / {e['kw_total']}** (10 catégories).")
    if un.get("comparison"):
        cmp = pd.DataFrame(un["comparison"]).sort_values("adjusted_rand_cat", ascending=False)
        lines = ["\n### 4.7 Non supervisé", "| Algorithme | Clusters | Silhouette | Davies-Bouldin | ARI | NMI | Pureté binaire |", "|---|---|---|---|---|---|---|"]
        for _, r in cmp.iterrows():
            lines.append(f"| {r.algorithm} | {r.n_clusters} | {r.silhouette:.3f} | {r.davies_bouldin:.3f} | {r.adjusted_rand_cat:.3f} | {r.nmi_cat:.3f} | {r.purity_binary:.3f} |")
        anom = pd.DataFrame(un["anomaly"]).sort_values("roc_auc", ascending=False)
        lines += ["\n| Détecteur d'anomalies | Apprentissage | ROC-AUC | F1 (seuil optimal) |", "|---|---|---|---|"]
        for _, r in anom.iterrows():
            lines.append(f"| {r.detector} | {r['mode']} | {r.roc_auc:.4f} | {r.best_f1:.4f} |")
        lines.append(f"\nApriori : {un.get('n_frequent_itemsets')} itemsets fréquents, {un.get('n_rules')} règles (confiance ≥ 0,7).")
        parts.append("\n".join(lines))
    if an.get("pcap_e2e"):
        lines = ["\n### 4.8 Chaîne complète PCAP → 47 features → modèle (captures avec vérité terrain)",
                 "| Capture | Profil | Flux évalués | Flux | Attaques | Rappel | Fausses alertes | F1 | Type d'attaque exact |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in an["pcap_e2e"]:
            lines.append(f"| {r['description']} | {r['profile']} | {r['subset']} | {r['n_flows']} | {r['n_attacks']} | {r['recall']:.3f} | "
                         f"{r['false_alarm_rate']:.3f} | {r['f1']:.3f} | {r['attack_type_accuracy']:.3f} |")
        lines.append("\n**Artefact TTL** : sur le même trafic réel rejoué avec des TTL de 64, le profil complet ne détecte presque plus rien, "
                     "alors que le profil portable garde les mêmes performances → profil portable recommandé pour une capture d'un autre réseau.")
        rt = an.get("replay_fidelity", [])
        if rt:
            lines.append("\nTest aller-retour (enregistrement réel → PCAP → features) : " +
                         ", ".join(f"`{r['feature']}` {r['concordance_pct']:.1f} %" for r in rt))
        fid = an.get("fidelity", [])
        if fid:
            lines.append("\nFidélité des formules de l'extracteur sur le CSV officiel : " +
                         ", ".join(f"`{r['feature']}` {r['concordance_pct']:.2f} %" for r in fid))
        parts.append("\n".join(lines))
    block = "<!-- RESULTS:START -->\n" + "\n".join(parts) + "\n<!-- RESULTS:END -->"
    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    txt = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, txt, flags=re.S)
    readme.write_text(txt, encoding="utf-8")
    print("README mis à jour.")


if __name__ == "__main__":
    main()
