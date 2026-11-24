# Division du projet — Network Intrusion Detection System

> Chaque membre commite **uniquement ses fichiers/dossiers**, avec **sa propre identité Git**.
> Personne ne modifie les fichiers d'un autre membre → aucun conflit.
> Cocher une case (`[x]`) quand le commit est **poussé** sur GitHub.

| Membre | Rôle | Commits prévus |
| --- | --- | --- |
| **Islem** (leader) | Notebooks, documentation `.md`, données, prétraitement, non supervisé, configuration | 9 |
| **Lina** | Modèles (entraînement, évaluation, prédiction) | 5 |
| **Zino** | Application web — backend (Flask, services) | 4 |
| **Amir** | Application web — frontend (pages HTML, CSS, JS) | 5 |
| **Sihem** | Extraction PCAP + documentation / rapport | 5 |

---

## 👑 Islem (leader)

- [x] **Commit 1 — structure** : `.gitignore`, dossiers vides (`.gitkeep`), `devision_project.md`
- [x] **Commit 2 — configuration** : `requirements.txt`, `pytest.ini`, `src/__init__.py`, `src/config.py`
- [x] **Commit 3 — nettoyage** : retrait de `notebooks/archive/`
- [x] **Commit 4 — données**
  - `data/raw/NUSW-NB15_features.csv`
  - `data/raw/UNSW-NB15_LIST_EVENTS.csv`
  - `data/interim/` *(dossier entier)*
  - `data/processed/` *(dossier entier)*
  - `src/prepare_data.py`
  - `scripts/download_data.py`
- [x] **Commit 5 — prétraitement & statistiques**
  - `src/preprocessing.py`
  - `src/features.py`
  - `src/visualization.py`
  - `src/evaluation/` *(dossier entier)*
  - `tests/test_core.py`
- [x] **Commit 6 — non supervisé**
  - `src/unsupervised/` *(dossier entier)*
  - `src/run_unsupervised.py`
- [x] **Commit 7 — notebooks**
  - `notebooks/01_eda_and_cleaning.ipynb`
  - `notebooks/02_preprocessing_encoding.ipynb`
  - `notebooks/03_model_training_evaluation.ipynb`
  - `notebooks/04_clustering.ipynb`
  - `notebooks/05_regles_association.ipynb`
  - `notebooks/06_explicabilite.ipynb`
  - `notebooks/07_extraction_pcap.ipynb`
  - `notebooks/build_notebooks.py`
- [x] **Commit 8 — résultats**
  - `reports/` *(dossier entier)*
  - `scripts/build_report.py`
  - `scripts/update_readme_results.py`
- [ ] **Commit 9 — documentation finale** *(en dernier, après tout le monde)*
  - `README.md`
  - `TEAM.md`
  - `AGENTS.md`

---

## Lina — Modèles

- [ ] **Commit 1 — entraînement**
  - `src/models.py`
  - `src/train.py`
  - `scripts/train.py`
- [ ] **Commit 2 — évaluation**
  - `src/evaluate.py`
  - `src/run_analysis.py`
- [ ] **Commit 3 — prédiction**
  - `src/predict.py`
  - `tests/test_predict.py`
- [ ] **Commit 4 — modèles déployés**
  - `models/binaire/` *(dossier entier)*
  - `models/multiclasse/` *(dossier entier)*
- [ ] **Commit 5 — tous les modèles comparés**
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

---

## Zino — Application (backend)

- [ ] **Commit 1 — services** : `src/services.py` *(plus gros fichier du projet)*
- [ ] **Commit 2 — routes Flask** : `src/app.py`
- [ ] **Commit 3 — point d'entrée** : `app.py` *(racine)*
- [ ] **Commit 4 — tests de l'application** : `tests/test_app.py`

---

## Amir — Application (frontend)

- [ ] **Commit 1 — gabarit & style**
  - `src/templates/base.html`
  - `src/static/` *(dossier entier)*
- [ ] **Commit 2 — accueil**
  - `src/templates/index.html`
  - `src/templates/about.html`
- [ ] **Commit 3 — pages d'analyse**
  - `src/templates/eda.html`
  - `src/templates/stats.html`
  - `src/templates/importance.html`
- [ ] **Commit 4 — pages modèles**
  - `src/templates/models.html`
  - `src/templates/predict.html`
- [ ] **Commit 5 — pages PCAP & non supervisé**
  - `src/templates/pcap.html`
  - `src/templates/extractor.html`
  - `src/templates/clustering.html`
  - `src/templates/rules.html`

---

## Sihem — Extraction PCAP & rapport

- [ ] **Commit 1 — lecture / écriture PCAP**
  - `src/pcap/__init__.py`
  - `src/pcap/reader.py`
  - `src/pcap/writer.py`
- [ ] **Commit 2 — reconstruction des flux**
  - `src/pcap/flows.py`
  - `src/pcap/app_layer.py`
- [ ] **Commit 3 — génération & rejeu**
  - `src/pcap/generator.py`
  - `src/pcap/replay.py`
- [ ] **Commit 4 — démos & tests**
  - `data/pcaps/` *(dossier entier)*
  - `scripts/predict_pcap.py`
  - `tests/test_pcap.py`
- [ ] **Commit 5 — rapport**
  - `docs/` *(dossier entier)*

---

## ❌ À ne JAMAIS commiter

Tout est déjà bloqué par le `.gitignore` :

- `zips/`, `zips.zip`
- `notebooks/.final_ipynb/`, `notebooks/archive/`, `notebooks/result_dict.pkl`
- `data/raw/UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` (dataset public → `scripts/download_data.py`)
- `data/.complete/`
- `__pycache__/`, `.pytest_cache/`
- `models/all/multiclass_full_voting.joblib`
- `models/all/multiclass_full_extra_trees.joblib`
- `models/all/multiclass_portable_random_forest.joblib`
- `NourMoustafathesis.pdf`, `Data_Mining_Course.pdf`, `mini_project_proposal (5).pdf`, `CLAUDE.md`, `PROMPTS.md`, `image*.png`

---

## Commandes (pour chaque membre)

```powershell
# une seule fois
git config user.name  "Prénom Nom"
git config user.email "email-du-compte-github"

# à chaque commit
git pull --rebase
git add <les fichiers du commit>
git status        # vérifier : aucun .zip, .pkl, ni fichier > 50 Mo
git commit -m "feat(pcap): ajout du reader et du writer"
git push
```
