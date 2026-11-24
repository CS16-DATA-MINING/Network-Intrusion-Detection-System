# Division du projet — Network Intrusion Detection System

> Chaque membre commite **uniquement ses fichiers/dossiers**, avec **sa propre identité Git**.
> Personne ne modifie les fichiers d'un autre membre → aucun conflit.

| Membre | Rôle | Commits prévus |
| --- | --- | --- |
| **Islem** (leader) | Notebooks, documentation `.md`, données, prétraitement, non supervisé, configuration | 8 |
| **Lina** | Modèles (entraînement, évaluation, prédiction) | 5 |
| **Zino** | Application web — backend (Flask, services) | 5 |
| **Amir** | Application web — frontend (pages HTML, CSS, JS) | 5 |
| **Sihem** | Extraction PCAP + documentation / rapport | 5 |

---

## 👑 Islem (leader)

### Dossiers entiers (Islem)

- `notebooks/archive/`
- `data/interim/`
- `data/processed/`
- `reports/`
- `src/evaluation/`
- `src/unsupervised/`

### Fichiers `notebooks/` (Islem)

- `notebooks/01_eda_and_cleaning.ipynb`
- `notebooks/02_preprocessing_encoding.ipynb`
- `notebooks/03_model_training_evaluation.ipynb`
- `notebooks/04_clustering.ipynb`
- `notebooks/05_regles_association.ipynb`
- `notebooks/06_explicabilite.ipynb`
- `notebooks/07_extraction_pcap.ipynb`
- `notebooks/build_notebooks.py`

### Fichiers `data/raw/` (Islem)

- `data/raw/.gitkeep`
- `data/raw/NUSW-NB15_features.csv`
- `data/raw/UNSW-NB15_LIST_EVENTS.csv`

### Fichiers racine (Islem)

- `.gitignore`, `requirements.txt`, `pytest.ini`
- `README.md`, `TEAM.md`, `AGENTS.md`, `devision_project.md`

### Fichiers `src/` (Islem)

- `src/__init__.py`, `src/config.py`
- `src/prepare_data.py`, `src/preprocessing.py`, `src/features.py`, `src/visualization.py`
- `src/run_unsupervised.py`

### Autres fichiers (Islem)

- `scripts/download_data.py`, `scripts/build_report.py`, `scripts/update_readme_results.py`
- `tests/test_core.py`

---

## Lina — Modèles

### Dossiers entiers (Lina)

- `models/binaire/`
- `models/multiclasse/`

### Fichiers `models/all/` (Lina)

- `models/all/registry.json`
- `models/all/linkages.joblib`
- `models/all/unsupervised.joblib`
- `models/all/binary_full_adaboost.joblib`
- `models/all/binary_full_bernoulli_nb.joblib`
- `models/all/binary_full_decision_tree.joblib`
- `models/all/binary_full_extra_trees.joblib`
- `models/all/binary_full_gaussian_nb.joblib`
- `models/all/binary_full_hist_gb.joblib`
- `models/all/binary_full_knn.joblib`
- `models/all/binary_full_lda.joblib`
- `models/all/binary_full_logistic_regression.joblib`
- `models/all/binary_full_multinomial_nb.joblib`
- `models/all/binary_full_random_forest.joblib`
- `models/all/binary_full_stacking.joblib`
- `models/all/binary_full_svm.joblib`
- `models/all/binary_full_voting.joblib`
- `models/all/binary_full_xgboost.joblib`
- `models/all/binary_full_zero_r.joblib`
- `models/all/binary_portable_decision_tree.joblib`
- `models/all/binary_portable_gaussian_nb.joblib`
- `models/all/binary_portable_knn.joblib`
- `models/all/binary_portable_random_forest.joblib`
- `models/all/binary_portable_xgboost.joblib`
- `models/all/binary_portable_zero_r.joblib`
- `models/all/multiclass_full_bernoulli_nb.joblib`
- `models/all/multiclass_full_decision_tree.joblib`
- `models/all/multiclass_full_gaussian_nb.joblib`
- `models/all/multiclass_full_hist_gb.joblib`
- `models/all/multiclass_full_knn.joblib`
- `models/all/multiclass_full_logistic_regression.joblib`
- `models/all/multiclass_full_multinomial_nb.joblib`
- `models/all/multiclass_full_random_forest.joblib`
- `models/all/multiclass_full_stacking.joblib`
- `models/all/multiclass_full_xgboost.joblib`
- `models/all/multiclass_full_zero_r.joblib`
- `models/all/multiclass_portable_decision_tree.joblib`
- `models/all/multiclass_portable_gaussian_nb.joblib`
- `models/all/multiclass_portable_knn.joblib`
- `models/all/multiclass_portable_xgboost.joblib`
- `models/all/multiclass_portable_zero_r.joblib`

### Fichiers `src/` (Lina)

- `src/models.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/run_analysis.py`

### Autres fichiers (Lina)

- `scripts/train.py`
- `tests/test_predict.py`

---

## Zino — Application (backend)

### Fichiers (Zino)

- `app.py` *(racine)*
- `src/app.py` — routes Flask
- `src/services.py` — logique métier de l'application (plus gros fichier du projet)
- `tests/test_app.py`

---

## Amir — Application (frontend)

### Dossiers entiers (Amir)

- `src/templates/`
- `src/static/`

---

## Sihem — Extraction PCAP & rapport

### Dossiers entiers (Sihem)

- `src/pcap/`
- `data/pcaps/`
- `docs/`

### Autres fichiers (Sihem)

- `scripts/predict_pcap.py`
- `tests/test_pcap.py`

---

## ❌ À ne JAMAIS commiter

Tout est déjà bloqué par le `.gitignore` :

- `zips/`, `zips.zip`
- `notebooks/.final_ipynb/`, `notebooks/result_dict.pkl`
- `data/raw/UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` (dataset public → `scripts/download_data.py`)
- `data/.complete/`
- `__pycache__/`, `.pytest_cache/`
- `models/all/multiclass_full_voting.joblib`
- `models/all/multiclass_full_extra_trees.joblib`
- `models/all/multiclass_portable_random_forest.joblib`
- `NourMoustafathesis.pdf`, `Data_Mining_Course.pdf`, `mini_project_proposal (5).pdf`, `CLAUDE.md`, `PROMPTS.md`, `image*.png`

---

## Ordre de travail

1. **Islem — commit 1** : `.gitignore` + dossiers vides (`.gitkeep`) + `devision_project.md` → push **avant tout le monde**.
2. **Islem — commits 2–3** : `requirements.txt`, `pytest.ini`, `src/__init__.py`, `src/config.py`, puis données + prétraitement
   (le reste du code en dépend).
3. **Les 4 autres, en parallèle** : `git pull`, copier ses fichiers, faire ~5 petits commits
   (ex. Amir : un commit par groupe de pages ; Lina : code → modèles déployés → `models/all` → tests).
4. **Islem — derniers commits** : notebooks, `reports/`, puis `README.md` final.

### Commandes (pour chaque membre)

```powershell
# une seule fois
git config user.name  "Prénom Nom"
git config user.email "email-du-compte-github"

# à chaque commit
git pull --rebase
git add <mes fichiers ou dossiers>
git status        # vérifier : aucun .zip, .pkl, ni fichier > 50 Mo
git commit -m "feat(pcap): ajout du reader et du writer"
git push
```
