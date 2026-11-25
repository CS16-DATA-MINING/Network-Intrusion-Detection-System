"""Services de l'application web : données, EDA filtrable, prédiction, PCAP.

Toute la logique métier est ici ; ``src/app.py`` ne contient que le routage.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src import config as C
from src.preprocessing import clean_dataframe, feature_frame, read_raw_file

# ---------------------------------------------------------------------------
# Chargement (mis en cache)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@lru_cache(maxsize=1)
def dataset() -> pd.DataFrame:
    if not C.CLEAN_FILE.exists():
        raise FileNotFoundError("data/processed/unsw_nb15_clean.csv absent : lancez `python -m src.prepare_data`.")
    df = pd.read_csv(C.CLEAN_FILE, low_memory=False)
    df["classe"] = np.where(df["label"] == 1, "Attaque", "Normal")
    return df


@lru_cache(maxsize=1)
def raw_sample() -> pd.DataFrame:
    """Échantillon brut (avant nettoyage) : sert à montrer les valeurs manquantes réelles."""
    if C.SAMPLE_FILE.exists():
        return pd.read_csv(C.SAMPLE_FILE, low_memory=False, nrows=40000)
    return read_raw_file(C.RAW_FILES[0], nrows=40000)


@lru_cache(maxsize=1)
def registry() -> dict:
    return _read_json(C.REGISTRY_FILE)


@lru_cache(maxsize=1)
def curves() -> dict:
    return _read_json(C.METRICS_DIR / "curves.json")


@lru_cache(maxsize=1)
def analysis() -> dict:
    return _read_json(C.METRICS_DIR / "analysis.json")


@lru_cache(maxsize=1)
def unsupervised() -> dict:
    return _read_json(C.METRICS_DIR / "unsupervised.json")


@lru_cache(maxsize=1)
def split_meta() -> dict:
    return _read_json(C.SPLIT_META_FILE)


@lru_cache(maxsize=1)
def unsup_models():
    p = C.ALL_MODELS_DIR / "unsupervised.joblib"
    return joblib.load(p) if p.exists() else None


@lru_cache(maxsize=1)
def linkages():
    p = C.ALL_MODELS_DIR / "linkages.joblib"
    return joblib.load(p) if p.exists() else None


def table(name: str) -> pd.DataFrame:
    p = C.METRICS_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def clean_json(obj):
    """Remplace NaN/inf par None (JSON valide)."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def records(df: pd.DataFrame) -> list[dict]:
    return clean_json(df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Synthèse générale
# ---------------------------------------------------------------------------
def overview() -> dict:
    meta, reg = split_meta(), registry()
    best = {}
    for profile in ("full", "portable"):
        for task in ("binary", "multiclass"):
            key = reg.get("best", {}).get(profile, {}).get(task)
            if not key:
                continue
            row = next((r for r in reg["metrics"][profile][task] if r["key"] == key), {})
            best[f"{profile}_{task}"] = {"key": key, "name": row.get("model"),
                                         "f1": row.get("f1"), "f1_macro": row.get("f1_macro"),
                                         "roc_auc": row.get("roc_auc"), "recall": row.get("recall"),
                                         "accuracy": row.get("accuracy"),
                                         "false_alarm_rate": row.get("false_alarm_rate")}
    n_models = sum(len(v) for p in reg.get("models", {}).values() for v in p.values())
    rs = meta.get("raw_stats", {})
    return clean_json({
        "raw_total": rs.get("full_total"), "raw_counts": rs.get("full_counts"),
        "files": rs.get("files"), "sample_total": rs.get("sample_total"),
        "sample_counts": rs.get("sample_counts"),
        "clean_total": meta.get("cleaning_report", {}).get("rows_out"),
        "cleaning": meta.get("cleaning_report"), "train_rows": meta.get("train_rows"),
        "test_rows": meta.get("test_rows"), "n_features": len(C.FEATURES),
        "n_raw_columns": len(C.RAW_COLUMNS), "n_models": n_models, "best": best,
        "train_counts": meta.get("train_counts"), "test_counts": meta.get("test_counts"),
    })


# ---------------------------------------------------------------------------
# Dashboard EDA (filtres interactifs)
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    cats = f.get("categories") or []
    if cats:
        df = df[df["attack_cat"].isin(cats)]
    for c in ("proto", "service", "state"):
        v = f.get(c)
        if v and v != "__all__":
            df = df[df[c] == v]
    lab = f.get("label")
    if lab in ("0", "1", 0, 1):
        df = df[df["label"] == int(lab)]
    var = f.get("range_var")
    if var in df.columns and f.get("range_min") not in (None, "") and f.get("range_max") not in (None, ""):
        df = df[(df[var] >= float(f["range_min"])) & (df[var] <= float(f["range_max"]))]
    return df


def eda_options() -> dict:
    df = dataset()
    return {"categories": C.ATTACK_CATEGORIES,
            "proto": df["proto"].value_counts().index[:25].tolist(),
            "service": df["service"].value_counts().index.tolist(),
            "state": df["state"].value_counts().index.tolist(),
            "numeric": C.NUMERIC_FEATURES, "categorical": C.CATEGORICAL_FEATURES + ["attack_cat", "classe"],
            "descriptions": C.FEATURE_DESCRIPTIONS,
            "ranges": {c: [float(df[c].min()), float(df[c].quantile(0.99)), float(df[c].max())]
                       for c in C.NUMERIC_FEATURES}}


def _hist_kde(x: np.ndarray, log: bool, bins: int = 40) -> dict:
    x = x[np.isfinite(x)]
    if log:
        x = np.log1p(np.clip(x, 0, None))
    if len(x) == 0:
        return {"edges": [], "counts": [], "kde_x": [], "kde_y": []}
    counts, edges = np.histogram(x, bins=bins)
    out = {"edges": edges.round(6).tolist(), "counts": counts.tolist(), "kde_x": [], "kde_y": []}
    if len(np.unique(x)) > 2 and len(x) > 5:
        xs = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
        try:
            kde = gaussian_kde(xs)
            grid = np.linspace(edges[0], edges[-1], 160)
            dens = kde(grid) * len(x) * (edges[1] - edges[0])   # mis à l'échelle des effectifs
            out["kde_x"], out["kde_y"] = grid.round(6).tolist(), dens.round(4).tolist()
        except (np.linalg.LinAlgError, ValueError):
            pass
    return out


def eda_payload(f: dict) -> dict:
    full = dataset()
    df = apply_filters(full, f)
    n = len(df)
    num = C.NUMERIC_FEATURES
    var = f.get("num_var") or "sbytes"
    cat_var = f.get("cat_var") or "proto"
    log = str(f.get("log", "true")).lower() == "true"
    method = f.get("corr_method") or "spearman"
    out = {"n_rows": n, "n_total": len(full), "n_attack": int(df["label"].sum()) if n else 0}
    if n == 0:
        return out
    # 1. statistiques descriptives
    d = df[num].describe(percentiles=[0.25, 0.5, 0.75]).T
    d["skew"] = df[num].skew()
    d["kurtosis"] = df[num].kurt()
    d["missing"] = df[num].isna().sum()
    d = d.reset_index().rename(columns={"index": "feature", "25%": "q1", "50%": "median", "75%": "q3"})
    out["describe"] = records(d.round(4))
    cat_desc = []
    for c in C.CATEGORICAL_FEATURES + ["attack_cat"]:
        vc = df[c].value_counts()
        cat_desc.append({"feature": c, "n_unique": int(df[c].nunique()), "mode": vc.index[0],
                         "mode_freq_pct": round(100 * vc.iloc[0] / n, 2)})
    out["describe_cat"] = cat_desc
    # 2. distribution numérique (globale et par classe)
    x = df[var].to_numpy(dtype=float)
    out["hist"] = {"var": var, "log": log, "all": _hist_kde(x, log)}
    for lab, name in ((0, "Normal"), (1, "Attaque")):
        xs = df.loc[df["label"] == lab, var].to_numpy(dtype=float)
        out["hist"][name] = _hist_kde(xs, log) if len(xs) else None
    # boîte par catégorie
    box = []
    for cat, g in df.groupby("attack_cat"):
        v = g[var].to_numpy(dtype=float)
        v = np.log1p(np.clip(v, 0, None)) if log else v
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        box.append({"cat": cat, "q1": q1, "median": med, "q3": q3,
                    "low": float(max(v.min(), q1 - 1.5 * iqr)), "high": float(min(v.max(), q3 + 1.5 * iqr)),
                    "n": int(len(v))})
    out["box"] = clean_json(box)
    # 3. fréquences catégorielles
    vc = df[cat_var].value_counts().head(15)
    split = pd.crosstab(df[cat_var], df["classe"]).reindex(vc.index).fillna(0)
    out["cat_freq"] = {"var": cat_var, "labels": vc.index.astype(str).tolist(), "counts": vc.tolist(),
                       "normal": split.get("Normal", pd.Series(0, index=vc.index)).astype(int).tolist(),
                       "attack": split.get("Attaque", pd.Series(0, index=vc.index)).astype(int).tolist()}
    # 4. corrélation
    cols = f.get("corr_vars") or ["dur", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "spkts",
                                  "dpkts", "smeansz", "dmeansz", "sjit", "djit", "tcprtt", "ct_srv_src",
                                  "ct_dst_ltm", "ct_src_ltm", "ct_dst_src_ltm", "ct_state_ttl", "label"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].sample(min(n, 20000), random_state=0) if n > 20000 else df[cols]
    corr = sub.corr(method=method).round(3).fillna(0)
    out["corr"] = {"method": method, "vars": cols, "matrix": corr.to_numpy().tolist()}
    # 5. composition
    out["composition"] = df["attack_cat"].value_counts().reindex(C.ATTACK_CATEGORIES).fillna(0).astype(int).to_dict()
    # 6. nuage de points (échantillon)
    yv = f.get("scatter_y") or "dbytes"
    s = df.sample(min(n, 2500), random_state=1)
    out["scatter"] = {"x_var": var, "y_var": yv, "x": s[var].round(4).tolist(),
                      "y": s[yv].round(4).tolist() if yv in s else [],
                      "label": s["label"].tolist(), "cat": s["attack_cat"].tolist()}
    return clean_json(out)


def missing_payload() -> dict:
    raw = raw_sample()
    raw = raw.rename(columns=lambda c: str(c).strip().lower())
    blank = raw.apply(lambda s: s.isna() | s.astype(str).str.strip().eq(""))
    pct = (blank.mean() * 100).round(3)
    clean = dataset()
    pct_clean = (clean[C.FEATURES].isna().mean() * 100).round(3)
    rep = split_meta().get("cleaning_report", {})
    return clean_json({"raw_pct": pct.to_dict(), "clean_pct": pct_clean.to_dict(),
                       "n_raw": int(len(raw)), "report": rep,
                       "explanations": {
                           "attack_cat": "Vide pour le trafic normal → remplacé par « Normal ».",
                           "ct_flw_http_mthd": "Vide hors HTTP dans les fichiers 2 à 4 → 0 (aucune méthode HTTP).",
                           "is_ftp_login": "Vide hors FTP → 0 ; 16 valeurs > 1 (erreur de codage) ramenées à 1.",
                           "ct_ftp_cmd": "Chaîne espace ' ' hors FTP → 0 (aucune commande).",
                       }})


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def models_payload(task: str, profile: str) -> dict:
    reg = registry()
    rows = reg.get("metrics", {}).get(profile, {}).get(task, [])
    cv = curves()
    an = analysis()
    out_curves = {}
    for r in rows:
        k = f"{task}|{profile}|{r['key']}"
        if k in cv:
            out_curves[r["key"]] = cv[k]
    cv_tables = {}
    for r in rows:
        t = table(f"cv_{task}_{profile}_{r['key']}.csv")
        if len(t):
            rank_cols = [c for c in t.columns if c.startswith("rank_test")]
            main = f"rank_test_{'f1' if task == 'binary' else 'f1_macro'}"
            sort_col = main if main in t.columns else (rank_cols[0] if rank_cols else t.columns[0])
            cv_tables[r["key"]] = records(t.sort_values(sort_col).head(25))
    fs = table(f"fold_scores_{task}_{profile}.csv")
    return clean_json({
        "rows": [{k: v for k, v in r.items() if k != "per_class"} for r in rows],
        "per_class": {r["key"]: r.get("per_class") for r in rows if r.get("per_class")},
        "curves": out_curves, "cv_tables": cv_tables,
        "best": reg.get("best", {}).get(profile, {}).get(task),
        "class_names": reg.get("class_names"),
        "fold_scores": fs.to_dict(orient="list") if len(fs) else {},
        "stats": an.get("model_stats", {}).get(task) if profile == "full" else None,
        "learning_curves": an.get("learning_curves") if task == "binary" and profile == "full" else None,
    })


def available_models(task: str = "binary", profile: str = "full") -> list[dict]:
    reg = registry()
    ms = reg.get("models", {}).get(profile, {}).get(task, {})
    # seuls les modèles dont le fichier est présent (vote / stacking non versionnés sur GitHub)
    return [{"key": k, "name": v["name"]} for k, v in ms.items()
            if k != "zero_r" and (C.ALL_MODELS_DIR / v["file"]).exists()]


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------
PRESETS = None


def presets() -> dict:
    """Exemples réels (médianes par catégorie du jeu de test) pour pré-remplir le formulaire."""
    global PRESETS
    if PRESETS is not None:
        return PRESETS
    test = pd.read_csv(C.TEST_FILE, low_memory=False)
    out = {}
    picks = [("Normal — HTTP", (test.attack_cat == "Normal") & (test.service == "http")),
             ("Normal — DNS", (test.attack_cat == "Normal") & (test.service == "dns")),
             ("Normal — FTP", (test.attack_cat == "Normal") & (test.service == "ftp")),
             ("Generic — rafale DNS", test.attack_cat == "Generic"),
             ("Exploits — HTTP", (test.attack_cat == "Exploits") & (test.service == "http")),
             ("Reconnaissance", test.attack_cat == "Reconnaissance"),
             ("DoS", test.attack_cat == "DoS"), ("Fuzzers", test.attack_cat == "Fuzzers"),
             ("Backdoor", test.attack_cat == "Backdoor"), ("Shellcode", test.attack_cat == "Shellcode"),
             ("Worms", test.attack_cat == "Worms")]
    for name, mask in picks:
        g = test[mask]
        if g.empty:
            continue
        # le flux le plus « typique » : le plus proche de la médiane (rangs)
        num = g[C.NUMERIC_FEATURES]
        z = (num.rank(pct=True) - 0.5).abs().sum(axis=1)
        row = g.loc[z.idxmin()]
        out[name] = {c: (row[c] if c in C.CATEGORICAL_FEATURES else float(row[c])) for c in C.FEATURES}
        out[name]["_true"] = row["attack_cat"]
    PRESETS = clean_json(out)
    return PRESETS


def predict_one(values: dict, model_key: str | None, profile: str, threshold: float) -> dict:
    from src.evaluation import explain as EX
    from src.models import load_model, predict_flows
    df = pd.DataFrame([{c: values.get(c) for c in C.FEATURES}])
    t0 = time.time()
    res = predict_flows(df, model_key=model_key, profile=profile, threshold=threshold)
    reg = registry()
    p_multi = None
    from src.models import predict_proba_frame
    pm, classes = predict_proba_frame(clean_dataframe(df), "multiclass", None, profile)
    p_multi = dict(zip(classes, pm[0].round(5).tolist()))
    xgb_pipe, _ = load_model("binary", "xgboost", profile)
    ex = EX.explain_instance(xgb_pipe, clean_dataframe(df), top=12)
    r = res.iloc[0]
    out = {"proba": float(r["proba_attaque"]), "prediction": int(r["prediction"]),
           "category": r["categorie_predite"], "category_confidence": float(r["confiance_categorie"]),
           "attack_type": r["categorie_attaque"], "attack_type_proba": float(r["proba_categorie_attaque"]),
           "multiclass": p_multi, "explanation": ex,
           "model": model_key or reg["best"][profile]["binary"],
           "latency_ms": round(1000 * (time.time() - t0), 1)}
    out["text"] = EX.text_explanation(out["prediction"], out["proba"], out["category"], ex["contributions"])
    um = unsup_models()
    if um is not None:
        Z = um["pca"].transform(um["prep"].transform(feature_frame(clean_dataframe(df))))
        cl = int(um["kmeans"].predict(Z)[0])
        score = float(-um["iforest"].score_samples(Z)[0])
        prof = next((p for p in um["cluster_profiles"] if int(p["cluster"]) == cl), {})
        out["cluster"] = {"id": cl, "profile": prof, "anomaly_score": score,
                          "anomaly": bool(score >= um["iforest_threshold"]),
                          "threshold": um["iforest_threshold"]}
    return clean_json(out)


# ---------------------------------------------------------------------------
# Analyse PCAP
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}


def analyze_capture(path: Path, filename: str, profile: str, model_key: str | None,
                    threshold: float, ct_mode: str, labels_path: Path | None = None) -> dict:
    from src.models import predict_flows
    from src.pcap import pcap_to_flows
    t0 = time.time()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        flows = pd.read_csv(path, low_memory=False)
        flows = flows.rename(columns=lambda c: str(c).strip().lower())
        n_packets = None
        if "srcip" not in flows.columns:
            for c, v in (("srcip", "?"), ("dstip", "?"), ("sport", 0), ("dsport", 0), ("stime", 0), ("ltime", 0)):
                flows[c] = v
    else:
        flows = pcap_to_flows(path, ct_mode=ct_mode)
        n_packets = flows.attrs.get("n_packets", 0)
    t_ext = time.time() - t0
    if flows.empty:
        raise ValueError("Aucun flux IP exploitable dans ce fichier (capture vide ou non IP).")
    from src.preprocessing import validate_features
    missing = validate_features(clean_dataframe(flows))
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier : {', '.join(missing)}")
    t1 = time.time()
    pred = predict_flows(flows, model_key=model_key, profile=profile, threshold=threshold)
    t_pred = time.time() - t1
    truth = None
    if labels_path is not None and labels_path.exists():
        from src.pcap.generator import match_ground_truth
        m = match_ground_truth(flows, pd.read_csv(labels_path))
        pred["true_attack_cat"] = m["true_attack_cat"].to_numpy()
        pred["true_label"] = m["true_label"].to_numpy()
        pred["true_in_train"] = m["true_in_train"].to_numpy()
    elif "attack_cat" in flows.columns or "label" in flows.columns:
        raw = clean_dataframe(flows)
        if "label" in raw:
            pred["true_label"] = raw["label"].to_numpy()
        if "attack_cat" in raw:
            pred["true_attack_cat"] = raw["attack_cat"].to_numpy()
    if "true_label" in pred:
        ok = pred["true_label"].notna().to_numpy()
        if ok.sum():
            from src.evaluate import binary_metrics
            y = pred.loc[ok, "true_label"].astype(int).to_numpy()
            truth = binary_metrics(y, pred.loc[ok, "prediction"].to_numpy(), pred.loc[ok, "proba_attaque"].to_numpy())
            if "true_attack_cat" in pred:
                att = ok & (pred["true_label"] == 1).to_numpy()
                truth["category_accuracy"] = float((pred.loc[ok, "categorie_predite"] == pred.loc[ok, "true_attack_cat"]).mean())
                truth["attack_type_accuracy"] = float((pred.loc[att, "categorie_predite"] == pred.loc[att, "true_attack_cat"]).mean()) if att.any() else None
            if "true_in_train" in pred and pred["true_in_train"].eq(False).any() and pred["true_in_train"].eq(True).any():
                un = ok & pred["true_in_train"].eq(False).to_numpy()
                yu = pred.loc[un, "true_label"].astype(int).to_numpy()
                bu = binary_metrics(yu, pred.loc[un, "prediction"].to_numpy(), pred.loc[un, "proba_attaque"].to_numpy())
                attu = un & (pred["true_label"] == 1).to_numpy()
                truth["unseen"] = {"n": int(un.sum()), "n_attacks": int(attu.sum()), "recall": bu["recall"],
                                   "false_alarm_rate": bu["false_alarm_rate"], "f1": bu["f1"],
                                   "attack_type_accuracy": float((pred.loc[attu, "categorie_predite"] == pred.loc[attu, "true_attack_cat"]).mean()) if attu.any() else None}
            cats = C.ATTACK_CATEGORIES
            if "true_attack_cat" in pred:
                ct = pd.crosstab(pred.loc[ok, "true_attack_cat"], pred.loc[ok, "categorie_predite"]).reindex(
                    index=cats, columns=cats).fillna(0).astype(int)
                truth["confusion_cat"] = {"labels": cats, "matrix": ct.to_numpy().tolist()}
    job = uuid.uuid4().hex[:12]
    pred.insert(0, "flow_id", np.arange(1, len(pred) + 1))
    JOBS[job] = {"pred": pred, "filename": filename, "created": time.time(), "profile": profile}
    for k in sorted(JOBS, key=lambda k: JOBS[k]["created"])[:-8]:      # garde 8 analyses
        JOBS.pop(k, None)
    return clean_json(capture_summary(job, pred, n_packets, t_ext, t_pred, truth, filename, profile, threshold))


def capture_summary(job, pred, n_packets, t_ext, t_pred, truth, filename, profile, threshold) -> dict:
    att = pred[pred["prediction"] == 1]
    cat_counts = pred["categorie_predite"].value_counts().reindex(C.ATTACK_CATEGORIES).fillna(0).astype(int)
    top_src = (att.groupby("srcip").agg(flux=("flow_id", "size"), proba_moy=("proba_attaque", "mean"),
                                        octets=("sbytes", "sum"),
                                        categories=("categorie_predite", lambda s: ", ".join(s.value_counts().index[:3])),
                                        destinations=("dstip", "nunique"))
               .sort_values(["flux", "proba_moy"], ascending=False).head(10).reset_index())
    top_dst = (att.groupby("dstip").agg(flux=("flow_id", "size"), sources=("srcip", "nunique"))
               .sort_values("flux", ascending=False).head(10).reset_index())
    t0 = float(pred["stime"].min()) if "stime" in pred and len(pred) else 0
    cols = ["flow_id", "srcip", "sport", "dstip", "dsport", "proto", "service", "state", "dur", "sbytes",
            "dbytes", "spkts", "dpkts", "sttl", "dttl", "proba_attaque", "prediction", "categorie_predite",
            "confiance_categorie", "categorie_attaque", "proba_categorie_attaque"] + [c for c in ("true_attack_cat",) if c in pred]
    return {
        "job": job, "filename": filename, "profile": profile, "threshold": threshold,
        "n_packets": n_packets, "n_flows": int(len(pred)), "n_attacks": int(len(att)),
        "pct_attacks": float(len(att) / len(pred)) if len(pred) else 0.0,
        "extraction_s": round(t_ext, 3), "prediction_s": round(t_pred, 3),
        "cat_counts": cat_counts.to_dict(),
        "proto_counts": pred["proto"].value_counts().head(8).to_dict(),
        "service_counts": pred["service"].value_counts().head(10).to_dict(),
        "top_src": records(top_src), "top_dst": records(top_dst),
        "timeline": {"t": (pred["stime"].astype(float) - t0).round(2).tolist() if "stime" in pred else [],
                     "proba": pred["proba_attaque"].round(4).tolist(),
                     "cat": pred["categorie_predite"].tolist(), "id": pred["flow_id"].tolist()},
        "flows": records(pred[[c for c in cols if c in pred]].round(5)),
        "truth": truth,
    }


def job_frame(job: str) -> pd.DataFrame:
    if job not in JOBS:
        raise KeyError("Analyse introuvable (expirée) : relancez l'analyse.")
    return JOBS[job]["pred"]


def explain_flow(job: str, flow_id: int) -> dict:
    from src.evaluation import explain as EX
    from src.models import load_model
    df = job_frame(job)
    row = df[df["flow_id"] == flow_id]
    if row.empty:
        raise KeyError("Flux introuvable")
    profile = JOBS[job]["profile"]
    pipe, _ = load_model("binary", "xgboost", profile)
    ex = EX.explain_instance(pipe, row[C.FEATURES], top=10)
    r = row.iloc[0]
    tuple_47 = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in C.RAW_COLUMNS[:-2] if c in row}
    return clean_json({"explanation": ex, "tuple": tuple_47,
                       "text": EX.text_explanation(int(r["prediction"]), float(r["proba_attaque"]),
                                                   r["categorie_predite"], ex["contributions"])})


# ---------------------------------------------------------------------------
# Dendrogramme (coordonnées pour Plotly)
# ---------------------------------------------------------------------------
def dendrogram_payload(method: str = "ward", p: int = 30) -> dict:
    from scipy.cluster.hierarchy import dendrogram
    lk = linkages()
    if lk is None or method not in lk:
        return {}
    d = dendrogram(lk[method], truncate_mode="lastp", p=p, no_plot=True)
    return clean_json({"icoord": d["icoord"], "dcoord": d["dcoord"], "labels": d["ivl"], "method": method})
