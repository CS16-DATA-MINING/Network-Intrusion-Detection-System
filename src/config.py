"""Configuration centrale du projet NIDS : chemins et constantes.

Le schéma des features est défini dans ``src/features.py`` et ré-exporté ici
(``C.FEATURES``, ``C.RAW_COLUMNS``…) pour un accès unique via ``src.config``.
"""
from __future__ import annotations

from pathlib import Path

from src.features import (ATTACK_CATEGORIES, CATEGORICAL_FEATURES,  # noqa: F401
                          EXCLUDED_NOISE, FEATURE_DESCRIPTIONS, FEATURE_GROUPS,
                          FEATURES, ID_COLUMNS, NUMERIC_FEATURES, RAW_COLUMNS,
                          TARGET_BINARY, TARGET_MULTI, TTL_FEATURES)

# ---------------------------------------------------------------------------
# Chemins (relatifs à la racine du dépôt, jamais absolus)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # CSV d'origine (lecture seule)
INTERIM_DIR = DATA_DIR / "interim"          # échantillon brut + jeu nettoyé
PROCESSED_DIR = DATA_DIR / "processed"      # train.csv / test.csv
SAMPLE_PCAP_DIR = DATA_DIR / "pcaps"        # PCAP de démonstration + vérité terrain

MODELS_DIR = ROOT / "models"
BINARY_MODEL_DIR = MODELS_DIR / "binaire"   # pipeline déployé (détection)
MULTI_MODEL_DIR = MODELS_DIR / "multiclasse"  # pipeline déployé (catégorie)
ALL_MODELS_DIR = MODELS_DIR / "all"         # tous les modèles comparés + registry.json

REPORTS_DIR = ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"       # tableaux CSV + JSON de résultats
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_FIG_DIR = FIGURES_DIR / "eda"
EVAL_FIG_DIR = FIGURES_DIR / "evaluation"
DOCS_DIR = ROOT / "docs"

# alias historiques (utilisés par le code d'analyse)
RESULTS_DIR = METRICS_DIR
TABLES_DIR = METRICS_DIR
REGISTRY_FILE = ALL_MODELS_DIR / "registry.json"

RAW_FILES = [RAW_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
RAW_FEATURES_FILE = RAW_DIR / "NUSW-NB15_features.csv"

SAMPLE_FILE = INTERIM_DIR / "unsw_nb15_sample_raw.csv"      # échantillon brut
CLEAN_FILE = INTERIM_DIR / "unsw_nb15_clean.csv"            # après nettoyage
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
SPLIT_META_FILE = PROCESSED_DIR / "split_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Échantillonnage stratifié par catégorie (le jeu complet = 2 540 047 flux).
# On garde toutes les lignes des classes rares (Worms, Shellcode, ...) et on
# plafonne les classes majoritaires.
SAMPLE_NORMAL = 60_000
SAMPLE_PER_ATTACK_CAP = 8_000


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for d in (INTERIM_DIR, PROCESSED_DIR, SAMPLE_PCAP_DIR, BINARY_MODEL_DIR, MULTI_MODEL_DIR,
              ALL_MODELS_DIR, METRICS_DIR, EDA_FIG_DIR, EVAL_FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
