"""Étape 1 : échantillonnage stratifié + nettoyage + split train/test stratifié.

Usage : ``python -m src.prepare_data``
Produit : data/processed/{unsw_nb15_sample_raw,unsw_nb15_clean,train,test}.csv
          et data/processed/split_metadata.json
"""
from __future__ import annotations

import time

from sklearn.model_selection import train_test_split

from src import config as C
from src.preprocessing import clean_training_data, load_raw_sample, save_json


def main() -> dict:
    C.ensure_dirs()
    missing = [p.name for p in C.RAW_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers bruts absents de data/raw : {missing}. "
            "Lancez d'abord `python scripts/download_data.py`.")
    t0 = time.time()
    sample, stats = load_raw_sample()
    sample.to_csv(C.SAMPLE_FILE, index=False)
    print(f"[1/3] Échantillon brut : {len(sample):,} lignes "
          f"(jeu complet : {stats['full_total']:,}) en {time.time() - t0:.0f}s")

    clean, rep = clean_training_data(sample)
    clean.to_csv(C.CLEAN_FILE, index=False)
    print(f"[2/3] Nettoyage : {rep['rows_in']:,} -> {rep['rows_out']:,} lignes "
          f"({rep['duplicates_removed']:,} doublons supprimés)")

    train, test = train_test_split(clean, test_size=C.TEST_SIZE,
                                   stratify=clean[C.TARGET_MULTI],
                                   random_state=C.RANDOM_STATE)
    train.to_csv(C.TRAIN_FILE, index=False)
    test.to_csv(C.TEST_FILE, index=False)
    meta = {
        "raw_stats": stats, "cleaning_report": rep,
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_size": C.TEST_SIZE, "random_state": C.RANDOM_STATE,
        "stratify": C.TARGET_MULTI,
        "train_counts": train[C.TARGET_MULTI].value_counts().to_dict(),
        "test_counts": test[C.TARGET_MULTI].value_counts().to_dict(),
    }
    save_json(meta, C.SPLIT_META_FILE)
    print(f"[3/3] Split stratifié : train={len(train):,} test={len(test):,}")
    return meta


if __name__ == "__main__":
    main()
