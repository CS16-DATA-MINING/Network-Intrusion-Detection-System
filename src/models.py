"""Catalogue des modèles, grilles d'hyperparamètres, chargement et prédiction.

Chaque modèle est un ``Pipeline(prep -> clf)`` : le préprocesseur est ré-ajusté
dans chaque pli de validation croisée (aucune fuite du pli de validation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config as C
from src.preprocessing import build_preprocessor, feature_frame

RS = C.RANDOM_STATE


@dataclass
class ModelSpec:
    key: str
    name: str
    family: str                       # famille pédagogique (cours / extension)
    estimator: Any
    grid: dict | list | None = None   # None = pas de réglage (Zero-R)
    search: str = "grid"              # grid | random
    n_iter: int = 15
    scaler: str = "standard"
    tune_subsample: int | None = None  # sous-échantillon pour la recherche
    fit_subsample: int | None = None   # sous-échantillon pour l'ajustement final
    tasks: tuple = ("binary", "multiclass")
    note: str = ""
    extra: dict = field(default_factory=dict)


def model_specs() -> list[ModelSpec]:
    """Liste ordonnée des modèles étudiés (cours obligatoire puis extensions)."""
    # profondeur bornée : limite la mémoire (8 Go) sans perte mesurable de F1
    tree_dist = {"clf__max_depth": [14, 20, 26],
                 "clf__min_samples_leaf": [1, 2, 4],
                 "clf__max_features": ["sqrt", 0.3, 0.5],
                 "clf__n_estimators": [100, 150, 200],
                 "clf__class_weight": [None, "balanced"]}
    return [
        ModelSpec("zero_r", "Zero-R (baseline)", "Cours — référence",
                  DummyClassifier(strategy="most_frequent"), None,
                  note="Prédit toujours la classe majoritaire (exempté de tuning)."),
        ModelSpec("knn", "K-NN", "Cours — paresseux",
                  KNeighborsClassifier(n_jobs=1),
                  {"clf__n_neighbors": [1, 3, 5, 9, 15],
                   "clf__weights": ["uniform", "distance"],
                   "clf__p": [1, 2]},
                  tune_subsample=20000,
                  note="Étude de K, distance (Manhattan p=1 / Euclidienne p=2), standardisation Z-score."),
        ModelSpec("decision_tree", "Arbre de décision (CART)", "Cours — arbres",
                  DecisionTreeClassifier(random_state=RS),
                  {"clf__criterion": ["gini", "entropy"],
                   "clf__max_depth": [8, 14, 20, None],
                   "clf__min_samples_leaf": [1, 5],
                   "clf__ccp_alpha": [0.0, 1e-4]},
                  note="Gini vs gain d'information (entropie), élagage coût-complexité (ccp_alpha)."),
        ModelSpec("random_forest", "Random Forest", "Cours — ensembles (bagging)",
                  RandomForestClassifier(random_state=RS, n_jobs=4), tree_dist,
                  search="random", n_iter=8),
        ModelSpec("gaussian_nb", "Naïve Bayes gaussien", "Cours — bayésien",
                  GaussianNB(), {"clf__var_smoothing": list(np.logspace(-12, -1, 12))}),
        ModelSpec("bernoulli_nb", "Naïve Bayes de Bernoulli", "Cours — bayésien",
                  BernoulliNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                                  "clf__binarize": [0.0, 0.5]},
                  note="Lissage de Laplace (alpha) ; variables binarisées autour de la moyenne."),
        ModelSpec("multinomial_nb", "Naïve Bayes multinomial", "Cours — bayésien",
                  MultinomialNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
                  scaler="minmax", note="Entrées Min-Max [0,1] (comptages positifs requis)."),
        ModelSpec("svm", "SVM (SVC)", "Cours — marges maximales",
                  SVC(probability=True, random_state=RS, cache_size=1000),
                  [{"clf__kernel": ["linear"], "clf__C": [0.1, 1]},
                   {"clf__kernel": ["rbf"], "clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.1]},
                   {"clf__kernel": ["poly"], "clf__C": [1, 10], "clf__degree": [2, 3], "clf__gamma": ["scale"]}],
                  tune_subsample=8000, fit_subsample=25000, tasks=("binary",),
                  note="Noyaux linéaire / RBF / polynomial ; complexité O(n²) -> réglage sur sous-échantillon."),
        ModelSpec("logistic_regression", "Régression logistique", "Extension — linéaire",
                  LogisticRegression(max_iter=3000),
                  {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]}),
        ModelSpec("lda", "Analyse discriminante linéaire", "Extension — linéaire",
                  LinearDiscriminantAnalysis(),
                  [{"clf__solver": ["svd"]},
                   {"clf__solver": ["lsqr"], "clf__shrinkage": [None, "auto", 0.1, 0.5]}],
                  tasks=("binary",)),
        ModelSpec("extra_trees", "Extra Trees", "Extension — ensembles",
                  ExtraTreesClassifier(random_state=RS, n_jobs=4), tree_dist,
                  search="random", n_iter=8),
        ModelSpec("adaboost", "AdaBoost", "Extension — boosting",
                  AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), random_state=RS),
                  {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.1, 0.5, 1.0],
                   "clf__estimator__max_depth": [1, 2]},
                  tasks=("binary",)),
        ModelSpec("hist_gb", "Gradient Boosting (histogrammes)", "Extension — boosting",
                  HistGradientBoostingClassifier(random_state=RS),
                  {"clf__learning_rate": loguniform(0.03, 0.3), "clf__max_iter": randint(150, 500),
                   "clf__max_leaf_nodes": randint(15, 63), "clf__l2_regularization": loguniform(1e-4, 1.0),
                   "clf__min_samples_leaf": randint(10, 60)},
                  search="random", n_iter=8),
        ModelSpec("xgboost", "XGBoost", "Extension — boosting",
                  XGBClassifier(random_state=RS, n_jobs=4, tree_method="hist", eval_metric="logloss"),
                  {"clf__n_estimators": randint(200, 600), "clf__max_depth": randint(4, 11),
                   "clf__learning_rate": loguniform(0.03, 0.3), "clf__subsample": uniform(0.7, 0.3),
                   "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 8),
                   "clf__reg_lambda": loguniform(0.1, 10)},
                  search="random", n_iter=10),
    ]


def make_pipeline(spec: ModelSpec, drop=()) -> Pipeline:
    from sklearn.base import clone
    return Pipeline([("prep", build_preprocessor(scaler=spec.scaler, drop=drop)),
                     ("clf", clone(spec.estimator))])


def make_ensembles(fitted: dict[str, Pipeline], task: str) -> dict[str, Pipeline]:
    """Ensembles construits à partir des meilleurs hyperparamètres trouvés."""
    from sklearn.base import clone
    keys = [k for k in ("random_forest", "extra_trees", "xgboost") if k in fitted]
    if len(keys) < 2:
        return {}
    clf = {k: clone(fitted[k].named_steps["clf"]) for k in keys}
    prep = build_preprocessor()
    voting = Pipeline([("prep", prep), ("clf", VotingClassifier(
        [(k, clf[k]) for k in keys], voting="soft", n_jobs=1))])
    stacking = Pipeline([("prep", build_preprocessor()), ("clf", StackingClassifier(
        [(k, clone(clf[k])) for k in keys], final_estimator=LogisticRegression(max_iter=2000),
        cv=3, n_jobs=1, stack_method="predict_proba"))])
    return {"voting": voting, "stacking": stacking}


ENSEMBLE_NAMES = {"voting": ("Vote souple (RF+ET+XGB)", "Extension — méta-ensemble"),
                  "stacking": ("Stacking (RF+ET+XGB → LR)", "Extension — méta-ensemble")}


# ---------------------------------------------------------------------------
# Chargement & prédiction (utilisé par l'application)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_registry() -> dict:
    path = C.REGISTRY_FILE
    if not path.exists():
        raise FileNotFoundError(
            "models/all/registry.json introuvable : lancez `python -m src.train`.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def load_model(task: str, key: str, profile: str = "full"):
    reg = load_registry()
    entry = reg["models"][profile][task][key]
    model = joblib.load(C.ALL_MODELS_DIR / entry["file"])
    encoder = None
    if task == "multiclass":
        encoder = reg["class_names"]
    return model, encoder


@lru_cache(maxsize=2)
def load_deployed(task: str = "binary"):
    """Pipeline déployé (models/binaire ou models/multiclasse) et ses métadonnées."""
    d = C.BINARY_MODEL_DIR if task == "binary" else C.MULTI_MODEL_DIR
    if not (d / "pipeline.joblib").exists():
        raise FileNotFoundError(f"{d.name}/pipeline.joblib introuvable : exécutez le notebook 03 "
                                "ou `python scripts/train.py`.")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return joblib.load(d / "pipeline.joblib"), meta


def predict_proba_frame(df: pd.DataFrame, task: str = "binary", key: str | None = None,
                        profile: str = "full") -> tuple[np.ndarray, list[str]]:
    """Retourne (probabilités, noms de classes) pour un DataFrame au schéma FEATURES.

    Sans ``key`` (profil complet), on utilise le pipeline déployé ; sinon le modèle
    demandé parmi ceux comparés (models/all/).
    """
    reg = load_registry()
    if key is None and profile == "full":
        model, meta = load_deployed(task)
        classes = meta["classes"]
    else:
        key = key or reg["best"][profile][task]
        model, classes = load_model(task, key, profile)
    X = feature_frame(df)
    proba = model.predict_proba(X)
    names = ["Normal", "Attaque"] if task == "binary" else list(classes)
    return proba, names


def predict_flows(df: pd.DataFrame, model_key: str | None = None, profile: str = "full",
                  threshold: float = 0.5, multiclass_key: str | None = None) -> pd.DataFrame:
    """Prédictions binaires + catégorie d'attaque pour chaque flux."""
    from src.preprocessing import clean_dataframe, validate_features
    df = clean_dataframe(df)
    missing = validate_features(df)
    if missing:
        raise ValueError(f"Features manquantes : {missing}")
    p_bin, _ = predict_proba_frame(df, "binary", model_key, profile)
    p_multi, classes = predict_proba_frame(df, "multiclass", multiclass_key, profile)
    out = df.copy()
    out["proba_attaque"] = p_bin[:, 1]
    out["prediction"] = (out["proba_attaque"] >= threshold).astype(int)
    idx = p_multi.argmax(axis=1)
    out["categorie_predite"] = [classes[i] for i in idx]
    out["confiance_categorie"] = p_multi.max(axis=1)
    # cohérence : un flux jugé normal par le détecteur binaire reste « Normal »
    out.loc[out["prediction"] == 0, "categorie_predite"] = "Normal"
    # type d'attaque le plus probable (hors « Normal ») : toujours fourni, pour que
    # chaque alerte — y compris après changement de seuil — ait une catégorie
    attack_cls = [i for i, c in enumerate(classes) if c != "Normal"]
    best_attack = np.array(attack_cls)[p_multi[:, attack_cls].argmax(axis=1)]
    p_att = p_multi[:, attack_cls]
    out["categorie_attaque"] = [classes[i] for i in best_attack]
    out["proba_categorie_attaque"] = p_att.max(axis=1) / np.clip(p_att.sum(axis=1), 1e-12, None)
    mask = (out["prediction"] == 1) & (out["categorie_predite"] == "Normal")
    out.loc[mask, "categorie_predite"] = out.loc[mask, "categorie_attaque"]
    out.loc[mask, "confiance_categorie"] = out.loc[mask, "proba_categorie_attaque"]
    return out
