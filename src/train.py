"""Entraînement, optimisation (CV 5-fold) et évaluation de tous les modèles.

Usage : ``python -m src.train [--fresh]`` (ou notebook 03, ou ``python scripts/train.py``)

Pour chaque tâche (binaire : normal/attaque ; multi-classe : 10 catégories) et
chaque profil de features (``full`` = 39 features ; ``portable`` = sans TTL) :
  1. GridSearchCV / RandomizedSearchCV, StratifiedKFold(5), scoring F1 (binaire)
     ou F1-macro (multi-classe), métriques secondaires enregistrées ;
  2. ré-ajustement de la meilleure configuration sur tout le train ;
  3. évaluation sur le jeu de test (jamais utilisé pour choisir un modèle :
     le meilleur modèle est sélectionné sur le score de CV) ;
  4. scores par pli sur des plis communs (tests de Friedman / Wilcoxon) ;
  5. sauvegarde : models/all/*.joblib + registry.json, reports/metrics/*.csv,
     reports/metrics/curves.json ; déploiement du meilleur pipeline dans
     models/binaire/ et models/multiclasse/ (pipeline.joblib + metadata.json).

Budgets de recherche volontairement réduits pour tenir sur 8 Go de RAM
(voir ``src/models.py``) ; parallélisme réglable par la variable NIDS_JOBS.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)

from src import config as C
from src.evaluate import (binary_metrics, confusion, curve_data,
                                    multiclass_metrics, scores_of, threshold_table)
from src.models import ENSEMBLE_NAMES, make_ensembles, make_pipeline, model_specs
from src.preprocessing import feature_frame, save_json

warnings.filterwarnings("ignore")

CLASS_NAMES = list(C.ATTACK_CATEGORIES)
PORTABLE_KEYS = ("zero_r", "decision_tree", "random_forest", "xgboost", "knn", "gaussian_nb")
COMMON_FOLD_N = 15000
# parallélisme (processus) : 2 par défaut pour tenir sur une machine de 8 Go
N_JOBS = int(os.environ.get("NIDS_JOBS", "2"))


def _subsample(X, y, n, seed=C.RANDOM_STATE):
    if n is None or len(X) <= n:
        return X, y
    from sklearn.model_selection import train_test_split
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def load_split():
    train = pd.read_csv(C.TRAIN_FILE, low_memory=False)
    test = pd.read_csv(C.TEST_FILE, low_memory=False)
    return train, test


def targets(df, task):
    if task == "binary":
        return df[C.TARGET_BINARY].astype(int).to_numpy()
    return df[C.TARGET_MULTI].map({c: i for i, c in enumerate(CLASS_NAMES)}).astype(int).to_numpy()


def run_task(task, profile, train, test, registry, curves, log=print):
    drop = C.TTL_FEATURES if profile == "portable" else ()
    Xtr, Xte = feature_frame(train), feature_frame(test)
    ytr, yte = targets(train, task), targets(test, task)
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "recall": "recall",
               "precision": "precision", "accuracy": "accuracy"} if task == "binary" else \
              {"f1_macro": "f1_macro", "accuracy": "accuracy",
               "balanced_accuracy": "balanced_accuracy", "f1_weighted": "f1_weighted"}
    refit = "f1" if task == "binary" else "f1_macro"
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    Xc, yc = _subsample(Xtr, ytr, COMMON_FOLD_N, seed=7)
    rows, fold_scores, fitted = [], {}, {}
    specs = [s for s in model_specs() if task in s.tasks]
    if profile == "portable":
        specs = [s for s in specs if s.key in PORTABLE_KEYS]
    for spec in specs:
        t0 = time.time()
        pipe = make_pipeline(spec, drop)
        best_params, cv_mean, cv_std, n_cfg = {}, np.nan, np.nan, 1
        if spec.grid is None:
            final = pipe
        else:
            Xt, yt = _subsample(Xtr, ytr, spec.tune_subsample)
            n_iter = max(4, spec.n_iter // 2) if profile == "portable" else spec.n_iter
            # parallélisme limité pour tenir dans 8 Go de RAM
            n_jobs = N_JOBS
            if spec.search == "grid":
                search = GridSearchCV(pipe, spec.grid, scoring=scoring, refit=refit, cv=cv,
                                      n_jobs=n_jobs, error_score="raise")
            else:
                search = RandomizedSearchCV(pipe, spec.grid, n_iter=n_iter, scoring=scoring,
                                            refit=refit, cv=cv, n_jobs=n_jobs,
                                            random_state=C.RANDOM_STATE, error_score="raise")
            search.fit(Xt, yt)
            best_params = {k.replace("clf__", ""): (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                           for k, v in search.best_params_.items()}
            res = pd.DataFrame(search.cv_results_)
            i = search.best_index_
            cv_mean, cv_std = res.loc[i, f"mean_test_{refit}"], res.loc[i, f"std_test_{refit}"]
            n_cfg = len(res)
            keep = [c for c in res.columns if c.startswith(("param_", "mean_test", "std_test", "rank_test", "mean_fit"))]
            res[keep].to_csv(C.METRICS_DIR / f"cv_{task}_{profile}_{spec.key}.csv", index=False)
            final = clone(pipe).set_params(**search.best_params_)
        t_search = time.time() - t0
        Xf, yf = _subsample(Xtr, ytr, spec.fit_subsample)
        t1 = time.time()
        final.fit(Xf, yf)
        t_fit = time.time() - t1
        if spec.grid is None:
            sc = cross_val_score(clone(pipe), Xtr, ytr, cv=cv, scoring=refit)
            cv_mean, cv_std = sc.mean(), sc.std()
        # scores par pli sur des plis communs (comparaison statistique)
        fold_scores[spec.key] = cross_val_score(clone(final), Xc, yc, cv=cv, scoring=refit,
                                                n_jobs=N_JOBS).tolist()
        t2 = time.time()
        y_pred = final.predict(Xte)
        t_pred = time.time() - t2
        sc = scores_of(final, Xte)
        rows.append(_evaluate(task, spec.key, spec.name, spec.family, yte, y_pred, sc,
                              best_params, cv_mean, cv_std, n_cfg, t_search, t_fit, t_pred,
                              spec.note, len(Xf), curves, profile))
        fitted[spec.key] = final
        fname = f"{task}_{profile}_{spec.key}.joblib"
        joblib.dump(final, C.ALL_MODELS_DIR / fname, compress=3)
        registry["models"].setdefault(profile, {}).setdefault(task, {})[spec.key] = {
            "file": fname, "name": spec.name, "family": spec.family, "params": best_params}
        log(f"  [{task}/{profile}] {spec.name:<34} CV {refit}={cv_mean:.4f}±{cv_std:.4f} "
            f"| test {rows[-1]['f1' if task == 'binary' else 'f1_macro']:.4f} "
            f"| {t_search + t_fit:.0f}s")

    if profile == "full":
        for key, pipe in make_ensembles(fitted, task).items():
            name, fam = ENSEMBLE_NAMES[key]
            t1 = time.time()
            pipe.fit(Xtr, ytr)
            t_fit = time.time() - t1
            sc_cv = cross_val_score(clone(pipe), Xc, yc, cv=cv, scoring=refit, n_jobs=N_JOBS)
            fold_scores[key] = sc_cv.tolist()
            t2 = time.time()
            y_pred = pipe.predict(Xte)
            t_pred = time.time() - t2
            rows.append(_evaluate(task, key, name, fam, yte, y_pred, scores_of(pipe, Xte), {},
                                  sc_cv.mean(), sc_cv.std(), 1, 0.0, t_fit, t_pred,
                                  "Combinaison des meilleurs RF, ET et XGBoost (CV sur plis communs).",
                                  len(Xtr), curves, profile))
            fname = f"{task}_{profile}_{key}.joblib"
            joblib.dump(pipe, C.ALL_MODELS_DIR / fname, compress=3)
            registry["models"][profile][task][key] = {"file": fname, "name": name,
                                                      "family": fam, "params": {}}
            log(f"  [{task}/{profile}] {name:<34} CV(commun) {sc_cv.mean():.4f} | test "
                f"{rows[-1]['f1' if task == 'binary' else 'f1_macro']:.4f}")

    table = pd.DataFrame(rows)
    table.drop(columns=["per_class"], errors="ignore").to_csv(
        C.METRICS_DIR / f"model_comparison_{task}_{profile}.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(C.METRICS_DIR / f"fold_scores_{task}_{profile}.csv", index=False)
    best = select_best(task, profile, registry)
    registry["metrics"].setdefault(profile, {})[task] = table.to_dict(orient="records")
    log(f"  => meilleur modèle {task}/{profile} (selon CV) : {best}")
    return table


def select_best(task: str, profile: str, registry: dict) -> str:
    """Choisit le modèle à déployer : meilleur score moyen sur les **plis communs**.

    Tous les modèles (y compris vote et stacking) sont évalués sur les mêmes 5 plis
    d'un même sous-échantillon stratifié : comparaison équitable, sans utiliser le test.
    (Le score de la recherche d'hyperparamètres n'est pas comparable entre modèles :
    il est calculé sur des données différentes et biaisé vers le haut par la sélection.)
    """
    fs = pd.read_csv(C.METRICS_DIR / f"fold_scores_{task}_{profile}.csv")
    common = fs.drop(columns=["zero_r"], errors="ignore").mean()
    best = str(common.idxmax())
    for r in registry["metrics"].get(profile, {}).get(task, []):
        if r["key"] in fs.columns:
            r["cv_common_mean"] = float(fs[r["key"]].mean())
            r["cv_common_std"] = float(fs[r["key"]].std(ddof=0))
    registry["best"].setdefault(profile, {})[task] = best
    return best


def _evaluate(task, key, name, family, yte, y_pred, sc, params, cv_mean, cv_std, n_cfg,
              t_search, t_fit, t_pred, note, n_fit, curves, profile):
    row = {"key": key, "model": name, "family": family, "cv_mean": float(cv_mean),
           "cv_std": float(cv_std), "n_configs": int(n_cfg), "search_time_s": round(t_search, 2),
           "fit_time_s": round(t_fit, 2), "predict_time_ms_per_1k": round(1000 * t_pred / len(yte) * 1000, 3),
           "n_train_fit": int(n_fit), "best_params": json.dumps(params, default=str), "note": note}
    ck = f"{task}|{profile}|{key}"
    if task == "binary":
        pos = None
        if sc is not None:
            pos = sc[:, 1] if np.ndim(sc) == 2 else sc
        m = binary_metrics(yte, y_pred, pos)
        row.update(m)
        row["test_main"] = m["f1"]
        curves[ck] = {"confusion": confusion(yte, y_pred, 2)}
        if pos is not None and np.std(pos) > 0:
            curves[ck].update(curve_data(yte, pos))
            if np.ndim(sc) == 2:
                curves[ck]["thresholds"] = threshold_table(yte, pos)
    else:
        proba = sc if (sc is not None and np.ndim(sc) == 2) else None
        m = multiclass_metrics(yte, y_pred, proba, CLASS_NAMES)
        row.update({k: v for k, v in m.items() if k != "per_class"})
        row["per_class"] = m["per_class"]
        row["test_main"] = m["f1_macro"]
        curves[ck] = {"confusion": confusion(yte, y_pred, len(CLASS_NAMES)),
                      "per_class": m["per_class"]}
    return row


CURVES_FILE = C.METRICS_DIR / "curves.json"
TASKS = [("binary", "full"), ("multiclass", "full"), ("binary", "portable"), ("multiclass", "portable")]


def start(resume: bool = True) -> tuple[dict, dict]:
    """Initialise (ou reprend) le registre des modèles et les données de courbes."""
    import sklearn
    import xgboost
    C.ensure_dirs()
    if resume and C.REGISTRY_FILE.exists() and CURVES_FILE.exists():
        return (json.loads(C.REGISTRY_FILE.read_text(encoding="utf-8")),
                json.loads(CURVES_FILE.read_text(encoding="utf-8")))
    registry = {"class_names": CLASS_NAMES, "features": C.FEATURES,
                "ttl_features": C.TTL_FEATURES, "models": {}, "best": {}, "metrics": {},
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "time_s": {},
                "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__}}
    return registry, {}


def train_task(task: str, profile: str, registry: dict, curves: dict, train=None, test=None,
               skip_done: bool = True, log=print) -> pd.DataFrame:
    """Recherche d'hyperparamètres + évaluation d'une (tâche, profil) ; sauvegarde immédiate."""
    base_log = log

    def log(msg):  # noqa: F811 — affichage + copie dans reports/metrics/train_log.txt
        base_log(msg)
        with open(C.METRICS_DIR / "train_log.txt", "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    if skip_done and registry.get("best", {}).get(profile, {}).get(task):
        log(f"[{task}/{profile}] déjà entraîné (reprise) — rechargement des résultats")
        return pd.read_csv(C.METRICS_DIR / f"model_comparison_{task}_{profile}.csv")
    if train is None:
        train, test = load_split()
    registry["train_rows"], registry["test_rows"] = int(len(train)), int(len(test))
    log(f"=== Tâche {task} — profil {profile} ===")
    t0 = time.time()
    table = run_task(task, profile, train, test, registry, curves, log=log)
    registry.setdefault("time_s", {})[f"{task}_{profile}"] = round(time.time() - t0, 1)
    save_json(registry, C.REGISTRY_FILE)
    CURVES_FILE.write_text(json.dumps(curves), encoding="utf-8")
    return table


def deploy(registry: dict) -> dict:
    """Copie le meilleur pipeline (profil complet, choisi sur la CV) dans
    models/binaire/ et models/multiclasse/ avec un fichier metadata.json."""
    import shutil
    out = {}
    for task, d in (("binary", C.BINARY_MODEL_DIR), ("multiclass", C.MULTI_MODEL_DIR)):
        key = select_best(task, "full", registry)
        entry = registry["models"]["full"][task][key]
        row = next(r for r in registry["metrics"]["full"][task] if r["key"] == key)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ALL_MODELS_DIR / entry["file"], d / "pipeline.joblib")
        meta = {"task": task, "model_key": key, "model_name": entry["name"], "profile": "full",
                "features": C.FEATURES, "categorical": C.CATEGORICAL_FEATURES,
                "classes": ["Normal", "Attaque"] if task == "binary" else CLASS_NAMES,
                "hyperparameters": entry["params"], "default_threshold": 0.5 if task == "binary" else None,
                "cv_score": row.get("cv_common_mean", row["cv_mean"]), "cv_std": row.get("cv_common_std", row["cv_std"]),
                "cv_search_score": row["cv_mean"],
                "test_metrics": {k: v for k, v in row.items() if isinstance(v, (int, float)) and k not in ("cv_mean", "cv_std")},
                "train_rows": registry.get("train_rows"), "test_rows": registry.get("test_rows"),
                "selection": "meilleur score moyen sur 5 plis stratifiés communs à tous les modèles (le test n'est pas utilisé)",
                "versions": registry.get("versions"), "deployed": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_json(meta, d / "metadata.json")
        out[task] = meta
        print(f"Déployé [{task}] : {entry['name']} -> {d.relative_to(C.ROOT) / 'pipeline.joblib'}")
    for task in ("binary", "multiclass"):        # même critère pour le profil portable
        if (C.METRICS_DIR / f"fold_scores_{task}_portable.csv").exists():
            select_best(task, "portable", registry)
    save_json(registry, C.REGISTRY_FILE)
    return out


def main(tasks=TASKS, resume: bool = True):
    registry, curves = start(resume)
    train, test = load_split()
    t0 = time.time()
    for task, profile in tasks:
        train_task(task, profile, registry, curves, train, test, skip_done=resume,
                   log=lambda s: print(s, flush=True))
    deploy(registry)
    print(f"Terminé en {(time.time() - t0) / 60:.1f} min")
    return registry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignorer les résultats existants (pas de reprise)")
    main(resume=not ap.parse_args().fresh)
