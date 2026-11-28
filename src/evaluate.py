"""Métriques d'évaluation (binaire et multi-classe) et données de courbes."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def scores_of(model, X) -> np.ndarray | None:
    """Scores continus pour ROC-AUC : predict_proba sinon decision_function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def binary_metrics(y_true, y_pred, score_pos) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),       # taux de détection
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,     # FPR
        "miss_rate": fn / (fn + tp) if (fn + tp) else 0.0,            # FNR
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if score_pos is not None and len(np.unique(y_true)) == 2 and np.std(score_pos) > 0:
        out["roc_auc"] = roc_auc_score(y_true, score_pos)
        out["pr_auc"] = average_precision_score(y_true, score_pos)
    else:
        out["roc_auc"] = 0.5
        out["pr_auc"] = float(np.mean(y_true))
    return {k: float(v) if not isinstance(v, int) else v for k, v in out.items()}


def multiclass_metrics(y_true, y_pred, proba, class_names) -> dict:
    labels = list(range(len(class_names)))
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if proba is not None and proba.ndim == 2 and proba.shape[1] == len(class_names):
        try:
            out["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                               average="macro", labels=labels)
        except ValueError:
            out["roc_auc_ovr"] = float("nan")
    else:
        out["roc_auc_ovr"] = float("nan")
    rep = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    out = {k: float(v) for k, v in out.items()}
    out["per_class"] = {c: {m: float(rep[c][m]) for m in ("precision", "recall", "f1-score", "support")}
                        for c in class_names}
    return out


def _thin(x, y, n=120):
    """Sous-échantillonne une courbe pour l'affichage (garde les extrémités)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) <= n:
        return x.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx].round(5).tolist(), y[idx].round(5).tolist()


def curve_data(y_true, score_pos) -> dict:
    fpr, tpr, _ = roc_curve(y_true, score_pos)
    prec, rec, _ = precision_recall_curve(y_true, score_pos)
    fx, fy = _thin(fpr, tpr)
    rx, ry = _thin(rec[::-1], prec[::-1])
    return {"roc": {"fpr": fx, "tpr": fy}, "pr": {"recall": rx, "precision": ry}}


def threshold_table(y_true, score_pos, thresholds=None) -> list[dict]:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = (score_pos >= t).astype(int)
        m = binary_metrics(y_true, pred, None)
        rows.append({"threshold": float(t), "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "false_alarm_rate": m["false_alarm_rate"],
                     "accuracy": m["accuracy"]})
    return rows


def confusion(y_true, y_pred, n_labels) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_labels))).tolist()
