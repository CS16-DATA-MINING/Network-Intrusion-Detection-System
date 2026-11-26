"""Ré-exécute l'entraînement du notebook 03 sans ouvrir Jupyter (même code : src/train.py).

    python scripts/train.py            # reprend là où l'entraînement s'est arrêté
    python scripts/train.py --fresh    # repart de zéro
    NIDS_JOBS=4 python scripts/train.py   # plus de parallélisme si ≥ 16 Go de RAM
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import main  # noqa: E402

if __name__ == "__main__":
    main(resume="--fresh" not in sys.argv)
