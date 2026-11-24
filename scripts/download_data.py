"""Téléchargement et vérification du jeu UNSW-NB15 (≈ 586 Mo, hors dépôt GitHub).

Usage :
    python scripts/download_data.py            # télécharge via l'API Kaggle puis vérifie
    python scripts/download_data.py --check    # vérifie seulement data/raw/

Sources :
  * officielle : https://research.unsw.edu.au/projects/unsw-nb15-dataset
    (dossier « CSV Files » : UNSW-NB15_1.csv … UNSW-NB15_4.csv, NUSW-NB15_features.csv)
  * miroir Kaggle : https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
    (nécessite `pip install kaggle` et un jeton ~/.kaggle/kaggle.json)

Téléchargement manuel : placer les 5 fichiers ci-dessus dans data/raw/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXPECTED = {  # nombre de lignes attendu par fichier
    "UNSW-NB15_1.csv": 700_001, "UNSW-NB15_2.csv": 700_001,
    "UNSW-NB15_3.csv": 700_001, "UNSW-NB15_4.csv": 440_044,
}
KAGGLE_SLUG = "mrwellsdavid/unsw-nb15"


def count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def check() -> bool:
    ok = True
    for name, n in EXPECTED.items():
        p = RAW / name
        if not p.exists():
            print(f"  ✗ {name} absent")
            ok = False
            continue
        got = count_lines(p)
        flag = "✓" if abs(got - n) <= 2 else "≠"
        ok &= flag == "✓"
        print(f"  {flag} {name} : {got:,} lignes (attendu {n:,})")
    feat = RAW / "NUSW-NB15_features.csv"
    print(f"  {'✓' if feat.exists() else '✗'} NUSW-NB15_features.csv (dictionnaire des 49 colonnes)")
    return ok and feat.exists()


def download_kaggle() -> None:
    if shutil.which("kaggle") is None:
        sys.exit("CLI kaggle introuvable : `pip install kaggle` puis placer kaggle.json, "
                 "ou télécharger manuellement (voir l'en-tête de ce script).")
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(RAW)], check=True)
    for z in RAW.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                name = Path(m).name
                if name in EXPECTED or name == "NUSW-NB15_features.csv":
                    with zf.open(m) as src, open(RAW / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        z.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="vérifier sans télécharger")
    args = ap.parse_args()
    if not args.check and not check():
        download_kaggle()
    print("Vérification de data/raw/ :")
    sys.exit(0 if check() else 1)
