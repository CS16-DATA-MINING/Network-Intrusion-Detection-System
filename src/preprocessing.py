"""Chargement, échantillonnage, nettoyage et prétraitement (pipeline scikit-learn).

Étapes :
1. ``load_raw_sample``   : lecture en flux des 4 CSV bruts UNSW-NB15 (2,54 M lignes)
                           et échantillonnage stratifié par catégorie d'attaque.
2. ``clean_dataframe``   : nettoyage *identique* pour le CSV et pour les flux
                           extraits d'un PCAP (types, valeurs manquantes, codes).
3. ``build_preprocessor``: ColumnTransformer ajusté UNIQUEMENT sur le train
                           (imputation, log1p data-driven, filtre de corrélation,
                           standardisation, One-Hot ``handle_unknown``).
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src import config as C

# ---------------------------------------------------------------------------
# 1. Chargement / échantillonnage
# ---------------------------------------------------------------------------
_CAT_FIX = {"Backdoors": "Backdoor", "": "Normal"}


def normalize_attack_cat(s: pd.Series) -> pd.Series:
    """NaN -> Normal, espaces supprimés, 'Backdoors' -> 'Backdoor'."""
    s = s.astype("object").where(s.notna(), "Normal").astype(str).str.strip()
    return s.replace(_CAT_FIX)


def read_raw_file(path, nrows: int | None = None) -> pd.DataFrame:
    """Lit un fichier brut UNSW-NB15 (sans en-tête) avec le schéma officiel."""
    return pd.read_csv(path, header=None, names=C.RAW_COLUMNS, nrows=nrows,
                       low_memory=False, encoding="latin1")


def load_raw_sample(n_normal: int = C.SAMPLE_NORMAL,
                    cap_per_attack: int = C.SAMPLE_PER_ATTACK_CAP,
                    random_state: int = C.RANDOM_STATE,
                    files: Iterable = C.RAW_FILES) -> tuple[pd.DataFrame, dict]:
    """Échantillon stratifié reproductible du jeu brut complet.

    Toutes les attaques sont conservées en mémoire (321 283 lignes) puis
    plafonnées par catégorie ; le trafic normal est d'abord sous-échantillonné
    fichier par fichier (pour limiter la mémoire) puis tiré uniformément.
    Retourne (échantillon, statistiques du jeu complet).
    """
    rng = np.random.default_rng(random_state)
    attacks, normals, stats = [], [], {"files": {}, "full_counts": {}}
    full_counts: dict[str, int] = {}
    for path in files:
        df = read_raw_file(path)
        # catégorie normalisée pour l'échantillonnage ; la colonne brute est conservée
        # telle quelle (valeurs vides, espaces) pour l'audit de qualité du notebook 01
        cat = normalize_attack_cat(df["attack_cat"])
        df["_cat"] = cat
        stats["files"][path.name] = int(len(df))
        for k, v in cat.value_counts().items():
            full_counts[k] = full_counts.get(k, 0) + int(v)
        is_norm = cat.eq("Normal")
        attacks.append(df[~is_norm])
        norm = df[is_norm]
        keep = rng.random(len(norm)) < 0.08          # pré-filtre mémoire (~8 %)
        normals.append(norm[keep])
        del df
    att = pd.concat(attacks, ignore_index=True)
    nor = pd.concat(normals, ignore_index=True)
    parts = [nor.sample(n=min(n_normal, len(nor)), random_state=random_state)]
    for cat, grp in att.groupby("_cat"):
        parts.append(grp.sample(n=min(cap_per_attack, len(grp)),
                                random_state=random_state))
    sample = pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=random_state).reset_index(drop=True)
    sample_counts = sample.pop("_cat").value_counts().to_dict()
    stats["full_counts"] = dict(sorted(full_counts.items(), key=lambda x: -x[1]))
    stats["full_total"] = int(sum(full_counts.values()))
    stats["sample_total"] = int(len(sample))
    stats["sample_counts"] = sample_counts
    return sample, stats


# ---------------------------------------------------------------------------
# 2. Nettoyage (partagé entre CSV et PCAP)
# ---------------------------------------------------------------------------
def _to_port(s: pd.Series) -> pd.Series:
    """Ports : entiers, hexadécimaux ('0x000b') ou '-' -> Int64 (NaN si inconnu)."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v in ("", "-"):
            return np.nan
        try:
            return int(v, 16) if v.lower().startswith("0x") else int(float(v))
        except ValueError:
            return np.nan
    return pd.to_numeric(s.map(conv), errors="coerce")


def clean_dataframe(df: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Nettoyage déterministe d'un DataFrame au schéma UNSW-NB15.

    - noms de colonnes normalisés (minuscules, sans espaces) ;
    - ports hexadécimaux / '-' convertis ;
    - ``ct_flw_http_mthd``, ``is_ftp_login``, ``ct_ftp_cmd`` : valeurs vides =
      flux non HTTP/FTP -> 0 (absence d'événement, pas une donnée inconnue) ;
    - ``is_ftp_login`` > 1 (erreur de codage connue) ramené à 1 ;
    - catégories textuelles en minuscules / sans espaces, vide -> '-' ;
    - valeurs infinies -> NaN (imputées ensuite par le pipeline).
    """
    rep = report if report is not None else {}
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "") for c in df.columns]
    for c in ("sport", "dsport"):
        if c in df:
            df[c] = _to_port(df[c])
    for c in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"):
        if c in df:
            raw = df[c]
            num = pd.to_numeric(raw.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}),
                                errors="coerce")
            rep[f"missing_{c}"] = int(num.isna().sum())
            df[c] = num.fillna(0)
    if "is_ftp_login" in df:
        rep["is_ftp_login_gt1"] = int((df["is_ftp_login"] > 1).sum())
        df["is_ftp_login"] = df["is_ftp_login"].clip(upper=1)
    for c in C.CATEGORICAL_FEATURES:
        if c in df:
            df[c] = (df[c].astype("object").where(df[c].notna(), "-")
                     .astype(str).str.strip().str.lower().replace({"": "-", "nan": "-"}))
    for c in C.NUMERIC_FEATURES:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = [c for c in C.NUMERIC_FEATURES if c in df]
    n_inf = int(np.isinf(df[num_cols].to_numpy(dtype=float)).sum()) if num_cols else 0
    rep["inf_values"] = n_inf
    if n_inf:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    if "attack_cat" in df:
        df["attack_cat"] = normalize_attack_cat(df["attack_cat"])
    if "label" in df:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


def clean_training_data(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoyage du jeu d'entraînement : clean_dataframe + doublons + cohérence."""
    rep: dict = {"rows_in": int(len(sample))}
    df = clean_dataframe(sample, rep)
    # cohérence label / attack_cat (label=0 <=> Normal)
    incoherent = (df["label"].eq(0) != df["attack_cat"].eq("Normal"))
    rep["incoherent_labels"] = int(incoherent.sum())
    df.loc[incoherent, "label"] = (~df.loc[incoherent, "attack_cat"].eq("Normal")).astype(int)
    # doublons exacts sur (features + cible) : sans identifiants ni horodatages
    key = C.FEATURES + [C.TARGET_MULTI]
    dup = df.duplicated(subset=key)
    rep["duplicates_removed"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    # vecteurs identiques étiquetés différemment (bruit d'étiquetage UNSW-NB15)
    rep["conflicting_vectors"] = int(df.duplicated(subset=C.FEATURES, keep=False).sum())
    rep["rows_out"] = int(len(df))
    rep["class_counts"] = df["attack_cat"].value_counts().to_dict()
    return df, rep


# ---------------------------------------------------------------------------
# 3. Transformateurs personnalisés (sérialisables avec joblib)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES = ["total_bytes", "total_pkts", "byte_ratio", "pkt_ratio",
                       "bytes_per_pkt"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute des features dérivées (calculables aussi depuis un PCAP)."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        sb, db = X["sbytes"].astype(float), X["dbytes"].astype(float)
        sp, dp = X["spkts"].astype(float), X["dpkts"].astype(float)
        X["total_bytes"] = sb + db
        X["total_pkts"] = sp + dp
        X["byte_ratio"] = sb / (db + 1.0)
        X["pkt_ratio"] = sp / (dp + 1.0)
        X["bytes_per_pkt"] = (sb + db) / (sp + dp + 1.0)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(base + ENGINEERED_FEATURES, dtype=object)


class SkewLog1p(BaseEstimator, TransformerMixin):
    """log1p appliqué aux colonnes dont l'asymétrie (train) dépasse ``threshold``.

    Justification : les tests de Shapiro-Wilk rejettent la normalité de toutes
    les variables ; les volumes (octets, débits) sont à queue lourde.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        sk = pd.DataFrame(X).skew().fillna(0).to_numpy()
        nonneg = np.nanmin(X, axis=0) >= 0
        self.log_mask_ = (np.abs(sk) > self.threshold) & nonneg
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        m = self.log_mask_
        X[:, m] = np.log1p(np.clip(X[:, m], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Supprime une variable de chaque paire |r de Pearson| > threshold (fit = train)."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr)
        n = X.shape[1]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if keep[j] and corr[i, j] > self.threshold:
                    keep[j] = False
        self.keep_mask_ = keep
        self.n_features_in_ = n
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_mask_]


# ---------------------------------------------------------------------------
# 4. Pipeline de prétraitement
# ---------------------------------------------------------------------------
def numeric_features(drop: Iterable[str] = ()) -> list[str]:
    drop = set(drop)
    return [c for c in C.NUMERIC_FEATURES if c not in drop] + ENGINEERED_FEATURES


def build_preprocessor(scaler: str = "standard", drop: Iterable[str] = (),
                       corr_threshold: float = 0.95) -> Pipeline:
    """Construit le pipeline de prétraitement (non ajusté).

    scaler : 'standard' (Z-score, défaut), 'minmax' (MultinomialNB : entrées
    positives requises) ou 'none'.
    drop   : features à exclure (ex. ``C.TTL_FEATURES`` pour le profil portable).
    """
    drop = list(drop)
    num_cols = numeric_features(drop)
    cat_cols = [c for c in C.CATEGORICAL_FEATURES if c not in drop]
    steps = [("impute", SimpleImputer(strategy="median")),
             ("log", SkewLog1p(threshold=2.0)),
             ("corr", CorrelationFilter(threshold=corr_threshold))]
    if scaler == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("scale", MinMaxScaler(clip=True)))
    num_pipe = Pipeline(steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="-")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=30, sparse_output=False)),
    ])
    ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                           verbose_feature_names_out=False)
    return Pipeline([("engineer", FeatureEngineer()), ("columns", ct)])


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les colonnes FEATURES (ajoute les absentes à NaN, ordre fixe)."""
    out = df.reindex(columns=C.FEATURES)
    for c in C.CATEGORICAL_FEATURES:
        out[c] = out[c].astype("object").where(out[c].notna(), "-").astype(str)
    for c in C.NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def validate_features(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des FEATURES manquantes (vide si tout est présent)."""
    return [c for c in C.FEATURES if c not in df.columns]


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
