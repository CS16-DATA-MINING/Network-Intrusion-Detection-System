# Guide d'équipe — organisation Git et répartition du travail

> Règle d'anonymat : uniquement les codes étudiants (CS16, CS10, CS26, CS27, RSD07) dans le code, les commits, le README et le rapport.
> La note Git (3 pts) est **individuelle** : un historique vide = 0/20 pour le membre concerné.

## Identité Git (une fois par membre)
```bash
git config user.name  "<votre code, ex. CS10>"
git config user.email "<adresse liée à votre compte GitHub>"
```

## Branches par périmètre
| Branche | Membre | Périmètre (fichiers) |
|---|---|---|
| `CS16-donnees-app` | CS16 (chef) | `src/prepare_data.py`, `scripts/download_data.py`, `src/app.py`, `src/services.py`, `src/templates/`, `src/static/`, `app.py`, `README.md` |
| `CS10-extraction-pcap` | CS10 | `src/pcap/`, `tests/test_pcap.py`, `notebooks/07_extraction_pcap.ipynb`, `data/pcaps/` |
| `CS26-eda-stats` | CS26 | `src/evaluation/statistical_tests.py`, `src/preprocessing.py`, `src/features.py`, notebooks 01–02, `reports/figures/eda/` |
| `CS27-modeles` | CS27 | `src/models.py`, `src/train.py`, `src/evaluate.py`, `src/evaluation/explain.py`, `src/run_analysis.py`, `src/predict.py`, notebooks 03 et 06, `models/`, `tests/test_predict.py` |
| `RSD07-non-supervise-rapport` | RSD07 | `src/unsupervised/`, `src/run_unsupervised.py`, notebooks 04–05, `scripts/build_report.py`, `docs/` |

Chaque membre commite **régulièrement** sur sa branche, ouvre une Pull Request vers `main`, et un autre membre la relit.

## Messages de commit
Format : `<type>(<périmètre>): <description>` — ex. `feat(pcap): calcul de ct_state_ttl (Algorithme 3.3)`,
`fix(app): gestion des fichiers PCAP vides`, `docs(readme): tableau des résultats`.

## Avant chaque Pull Request
```bash
python -m pytest                 # tous les tests passent
python app.py                    # l'application démarre, les pages s'affichent
```
