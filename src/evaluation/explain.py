"""Explicabilité : importances globales et explications locales (SHAP / TreeSHAP).

Les valeurs SHAP sont calculées nativement par XGBoost
(``Booster.predict(pred_contribs=True)``, algorithme TreeSHAP exact) : aucune
dépendance supplémentaire. Les contributions des colonnes One-Hot sont
ré-agrégées sur la variable d'origine (ex. ``service_http`` -> ``service``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config as C
from src.preprocessing import feature_frame


def transformed_names(pipe) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def raw_feature_of(name: str) -> str:
    for c in C.CATEGORICAL_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


def aggregate_to_raw(names, values) -> pd.Series:
    s = pd.Series(values, index=[raw_feature_of(n) for n in names])
    return s.groupby(level=0).sum()


def shap_values(pipe, df: pd.DataFrame) -> tuple[np.ndarray, list[str], float | np.ndarray]:
    """SHAP (log-odds) pour un pipeline dont l'étape finale est un XGBClassifier.

    Retourne (contributions [n, p] ou [n, classes, p], noms transformés, biais).
    """
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Z = prep.transform(feature_frame(df))
    contrib = clf.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
    names = transformed_names(pipe)
    if contrib.ndim == 3:            # multi-classe : [n, classes, p+1]
        return contrib[:, :, :-1], names, contrib[:, :, -1]
    return contrib[:, :-1], names, contrib[:, -1]


def explain_instance(pipe, row: pd.DataFrame, class_index: int | None = None,
                     top: int = 10) -> dict:
    """Explication locale d'un flux : contributions SHAP agrégées par feature brute."""
    vals, names, bias = shap_values(pipe, row)
    if vals.ndim == 3:
        ci = class_index if class_index is not None else int(np.argmax(vals[0].sum(axis=1) + bias[0]))
        v, b = vals[0, ci], float(bias[0, ci])
    else:
        v, b = vals[0], float(bias[0])
    agg = aggregate_to_raw(names, v)
    order = agg.abs().sort_values(ascending=False).index[:top]
    # valeurs affichées : features d'origine + features dérivées (calculées comme dans le pipeline)
    frame = pipe.named_steps["prep"].named_steps["engineer"].transform(feature_frame(row))
    items = []
    for f in order:
        val = frame.iloc[0][f] if f in frame.columns else None
        items.append({"feature": f, "value": None if val is None else (val if isinstance(val, str) else float(val)),
                      "contribution": float(agg[f]),
                      "description": C.FEATURE_DESCRIPTIONS.get(f, "Feature dérivée")})
    logit = b + float(v.sum())
    return {"base_logit": b, "logit": logit, "probability": float(1 / (1 + np.exp(-logit))),
            "contributions": items}


def global_shap(pipe, df: pd.DataFrame) -> pd.Series:
    vals, names, _ = shap_values(pipe, df)
    if vals.ndim == 3:
        vals = np.abs(vals).sum(axis=1)
    return aggregate_to_raw(names, np.abs(vals).mean(axis=0)).sort_values(ascending=False)


def text_explanation(pred_label: int, proba: float, category: str, contribs: list[dict]) -> str:
    """Synthèse textuelle déterministe (pas de LLM) d'une prédiction."""
    verdict = "une ATTAQUE" if pred_label else "du trafic NORMAL"
    conf = "très élevée" if abs(proba - 0.5) > 0.45 else "élevée" if abs(proba - 0.5) > 0.3 else "modérée"
    lines = [f"Le modèle classe ce flux comme {verdict} (probabilité d'attaque {proba:.1%}, "
             f"confiance {conf})."]
    if pred_label:
        lines.append(f"Catégorie d'attaque la plus probable : **{category}**.")
    up = [c for c in contribs if c["contribution"] > 0][:3]
    down = [c for c in contribs if c["contribution"] < 0][:3]

    def fmt(c):
        v = c["value"]
        v = v if isinstance(v, str) else (f"{v:.4g}" if v is not None else "?")
        return f"{c['feature']} = {v} ({c['description']})"
    if up:
        lines.append("Éléments qui poussent vers « attaque » : " + " ; ".join(fmt(c) for c in up) + ".")
    if down:
        lines.append("Éléments qui poussent vers « normal » : " + " ; ".join(fmt(c) for c in down) + ".")
    return " ".join(lines)
