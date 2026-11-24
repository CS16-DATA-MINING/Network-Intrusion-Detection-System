# 🛡️ NIDS UNSW-NB15 — Détection d'intrusions réseau, du PCAP à la décision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-green)
![Flask](https://img.shields.io/badge/Flask-web%20app-lightgrey?logo=flask)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

> Mini-projet Data Mining 2026/2027 — **Parcours hybride** (classification supervisée + clustering non supervisé + règles d'association).
> Équipe identifiée uniquement par codes étudiants (règle d'anonymat).

---

## Sommaire
1. [Présentation, objectifs et description du projet](#1-présentation-objectifs-et-description-du-projet)
2. [Installation, exécution locale et accès aux données](#2-installation-exécution-locale-et-accès-aux-données)
3. [Rôles des membres et matrice des contributions](#3-rôles-des-membres-et-matrice-des-contributions)
4. [Résultats clés et synthèse des découvertes](#4-résultats-clés-et-synthèse-des-découvertes)
5. [Architecture du dépôt](#5-architecture-du-dépôt) · [Application web](#6-application-web) · [Limites](#7-limites-et-perspectives)

---

## 1. Présentation, objectifs et description du projet

### Contexte métier
Un **système de détection d'intrusions réseau (NIDS)** analyse le trafic pour signaler les flux malveillants
(scans, dénis de service, exploits, vers…) noyés dans un trafic légitime ultra-majoritaire. Le coût d'une attaque
manquée étant bien supérieur à celui d'une fausse alerte, la métrique prioritaire est le **rappel des attaques**,
sous contrainte d'un taux de fausses alertes faible.

### Objectifs
| Objectif scientifique | Objectif applicatif |
|---|---|
| Comparer 15 classifieurs (cours + extensions) avec optimisation systématique en CV 5-fold | Application web 100 % locale : EDA interactive, prédiction temps réel, analyse de fichiers PCAP |
| Justifier statistiquement les choix (6 familles de tests, correction BH) | **Extraire les 47 features UNSW-NB15 depuis n'importe quelle capture** (réimplémentation Argus + Bro-IDS) |
| Découvrir des structures sans étiquettes (7 algorithmes de clustering, détection d'anomalies) | Expliquer chaque alerte (SHAP local + texte) |
| Extraire des signatures lisibles (Apriori) | API REST (`/api/predict`, `/api/pcap`) |

### Le jeu de données UNSW-NB15
| | |
|---|---|
| **Source** | UNSW Canberra — N. Moustafa & J. Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS 2015 ; thèse de N. Moustafa (2017) |
| **Lien** | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| **Volume brut** | **2 540 047 enregistrements** (4 fichiers CSV, ≈ 586 Mo) — ≫ 10 000 exigés |
| **Attributs** | **49 colonnes** (47 features + `attack_cat` + `label`) — ≫ 10 exigés |
| **Génération** | 100 Go de trafic PCAP (IXIA PerfectStorm), features produites par Argus, Bro-IDS et 12 algorithmes additionnels |
| **Cibles** | `label` (0 normal / 1 attaque) ; `attack_cat` : Normal + 9 familles (Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) |
| **Déséquilibre** | 87,4 % normal ; *Worms* = 174 flux |

**Pourquoi ce choix ?** UNSW-NB15 est un jeu de référence moderne (contrairement à KDD99), et sa thèse
d'origine documente précisément **comment les features sont générées à partir des paquets** : nous avons pu
réimplémenter cette chaîne pour appliquer les modèles à de vraies captures (`src/pcap/`).

### Pipeline de bout en bout
```
PCAP/PCAPNG ──► src/pcap/reader.py     (décodage Ethernet/VLAN/SLL → IPv4/IPv6 → TCP/UDP/ICMP/ARP)
            ──► src/pcap/flows.py      (flux bidirectionnels « Argus » : état, octets, TTL, pertes, gigue, RTT)
            ──► src/pcap/app_layer.py  (« Bro-IDS » : service par signature, HTTP, FTP)
            ──► Algorithmes 3.2–3.9    (is_sm_ips_ports, ct_state_ttl, ct_* sur 100 connexions)
            ──► 47 features UNSW-NB15  ──► src/preprocessing.py (pipeline ajusté sur le train)
            ──► src/models.py          (meilleur modèle CV) ──► alerte + catégorie + explication SHAP
```

---

## 2. Installation, exécution locale et accès aux données

### Prérequis
Python ≥ 3.10 (testé en 3.13), 8 Go de RAM, ≈ 1,5 Go d'espace disque.

### Étape par étape
```bash
# 1. Cloner le dépôt
git clone <URL-du-dépôt> && cd Network-Intrusion-Detection-System

# 2. Environnement et dépendances
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt

# 3. Données brutes (> 100 Mo : stockage externe, non versionnées)
python scripts/download_data.py   # via l'API Kaggle, ou téléchargement manuel (voir ci-dessous)

# 4. Préparation : échantillonnage stratifié, nettoyage, split 80/20 (~20 s)
python -m src.prepare_data

# 5. Notebooks dans l'ordre — l'ENTRAÎNEMENT se fait dans le notebook 03
#    (recherches CV 5 plis, binaire + multi-classe, sauvegarde joblib + metadata.json)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/0*.ipynb
#    équivalent sans Jupyter :  python scripts/train.py   (~1 h ; NIDS_JOBS=4 si ≥ 16 Go de RAM)
#    puis : python -m src.run_unsupervised  et  python -m src.run_analysis

# 6. Prédiction en ligne de commande sur une capture
python scripts/predict_pcap.py data/pcaps/demo_unsw_testbed.pcap -o predictions.csv

# 7. Lancer l'application web
python app.py                      # puis ouvrir http://127.0.0.1:5000
```

### Accès aux données (stockage externe)
Le jeu brut (≈ 586 Mo) dépasse la limite GitHub ; il n'est **pas versionné** :
- **Officiel** : https://research.unsw.edu.au/projects/unsw-nb15-dataset → dossier *CSV Files* ;
- **Miroir Kaggle** : `kaggle datasets download -d mrwellsdavid/unsw-nb15` (utilisé par `scripts/download_data.py`).

Placer `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` et `NUSW-NB15_features.csv` dans `data/raw/`, puis
`python scripts/download_data.py --check` vérifie le nombre de lignes de chaque fichier.

Les **captures PCAP de démonstration** (avec vérité terrain par flux) sont versionnées dans `data/pcaps/` :
- `demo_unsw_replay.pcap` (9 Mo) : **rejeu de 2 496 enregistrements réels** UNSW-NB15 (9 familles d'attaques),
  et `demo_unsw_replay_ttl64.pcap`, le même trafic avec des TTL d'hôtes modernes — `python -m src.pcap.replay` ;
- `demo_unsw_testbed.pcap`, `demo_modern_ttl.pcap` : scénarios écrits à la main — `python -m src.pcap.generator`.

### Tests
```bash
python -m pytest            # extracteur PCAP, prétraitement, algorithmes implémentés, prédiction, application
```

### Notebooks (dans l'ordre)
| Notebook | Contenu |
|---|---|
| `01_eda_and_cleaning.ipynb` | chargement, échantillonnage, qualité, nettoyage, EDA, 6 familles de tests statistiques |
| `02_preprocessing_encoding.ipynb` | split stratifié, pipeline (log1p, filtre de corrélation, Z-score, One-Hot), anti-fuite |
| `03_model_training_evaluation.ipynb` | **entraînement** des 15 modèles (GridSearch/RandomizedSearch CV 5-fold), sauvegarde des pipelines, métriques, ROC/PR, Friedman/McNemar |
| `04_clustering.ipynb` | K-Means, PAM, CLARA, AGNES, DIANA, DBSCAN, GMM, détection d'anomalies |
| `05_regles_association.ipynb` | Apriori, règles ⇒ classe / catégorie |
| `06_explicabilite.ipynb` | 5 méthodes d'importance, SHAP global/local, ablation |
| `07_extraction_pcap.ipynb` | génération des tuples depuis un PCAP, validation des formules, évaluation bout-en-bout |

Les notebooks sont générés par `notebooks/build_notebooks.py` et appellent le code de `src/` (aucune duplication).

---

## 3. Rôles des membres et matrice des contributions

> Règle d'anonymat : uniquement les codes étudiants (aucun nom réel). Équipe de 5 membres.

| Code | Rôle | Responsabilités principales | Fichiers / notebooks |
|---|---|---|---|
| **CS16** (chef d'équipe) | Coordination, données & application | Organisation du dépôt, téléchargement et échantillonnage stratifié, nettoyage, split ; application web Flask et API REST | `src/prepare_data.py`, `scripts/download_data.py`, `src/app.py`, `src/services.py`, `src/templates/*`, `src/static/*` |
| **CS10** | Extraction PCAP | Réimplémentation Argus / Bro-IDS, features additionnelles (Alg. 3.2–3.9), rejeu et test aller-retour, générateur de captures | `src/pcap/*`, `tests/test_pcap.py`, NB 07 |
| **CS26** | EDA & statistiques | Visualisations, 6 familles de tests statistiques, prétraitement et encodage | `src/evaluation/statistical_tests.py`, `src/preprocessing.py`, `src/features.py`, NB 01–02 |
| **CS27** | Modélisation & explicabilité | 15 modèles, optimisation CV 5 plis, évaluation, comparaison statistique, importance des variables, SHAP | `src/models.py`, `src/train.py`, `src/evaluate.py`, `src/evaluation/explain.py`, `src/run_analysis.py`, NB 03 et 06 |
| **RSD07** | Non supervisé & rapport | Clustering (K-Means, PAM, CLARA, AGNES, DIANA, DBSCAN, GMM), anomalies, Apriori, rapport PDF | `src/unsupervised/*`, `src/run_unsupervised.py`, `scripts/build_report.py`, NB 04–05 |

| Tâche | CS16 | CS10 | CS26 | CS27 | RSD07 |
|---|:-:|:-:|:-:|:-:|:-:|
| Coordination & structure du dépôt | ●● | | | | |
| Données & nettoyage | ●● | | ● | | |
| Extracteur PCAP (47 features) | | ●● | | | |
| EDA & tests statistiques | | | ●● | | |
| Prétraitement | | | ●● | ● | |
| Modèles supervisés & tuning | | | | ●● | |
| Explicabilité (SHAP, importances) | | | | ●● | ● |
| Clustering, anomalies, Apriori | | | | | ●● |
| Application web & API | ●● | ● | | | |
| Tests (pytest) | ● | ●● | | ● | |
| Rapport PDF & README | ● | ● | ● | ● | ●● |

---

## 4. Résultats clés et synthèse des découvertes

<!-- RESULTS:START -->
### 4.1 Détection binaire (normal / attaque) — jeu de test (18 258 flux), profil complet
| Modèle | Famille | F1 CV plis communs ★ | F1 CV recherche | Accuracy | Précision | Rappel | **F1** | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|---|---|
| **SVM (SVC)** 🏆 | Cours — marges maximales | 0.9867 | 0.9899 | 99.10 % | 97.65 % | 99.99 % | **98.80 %** | 0.9971 | 1.43 % |
| Vote souple (RF+ET+XGB) | Extension — méta-ensemble | 0.9866 | 0.9866 | 99.10 % | 97.65 % | 99.99 % | **98.80 %** | 0.9990 | 1.43 % |
| Extra Trees | Extension — ensembles | 0.9865 | 0.9875 | 99.10 % | 97.66 % | 99.97 % | **98.80 %** | 0.9988 | 1.42 % |
| Gradient Boosting (histogrammes) | Extension — boosting | 0.9857 | 0.9874 | 99.09 % | 97.67 % | 99.94 % | **98.79 %** | 0.9990 | 1.41 % |
| Arbre de décision (CART) | Cours — arbres | 0.9825 | 0.9873 | 99.09 % | 97.63 % | 99.97 % | **98.79 %** | 0.9976 | 1.44 % |
| XGBoost | Extension — boosting | 0.9857 | 0.9873 | 99.09 % | 97.64 % | 99.96 % | **98.79 %** | 0.9991 | 1.43 % |
| Random Forest | Cours — ensembles (bagging) | 0.9862 | 0.9875 | 99.09 % | 97.64 % | 99.96 % | **98.79 %** | 0.9988 | 1.43 % |
| Régression logistique | Extension — linéaire | 0.9859 | 0.9873 | 99.08 % | 97.64 % | 99.94 % | **98.78 %** | 0.9981 | 1.43 % |
| AdaBoost | Extension — boosting | 0.9859 | 0.9870 | 99.05 % | 97.59 % | 99.91 % | **98.74 %** | 0.9986 | 1.47 % |
| Stacking (RF+ET+XGB → LR) | Extension — méta-ensemble | 0.9860 | 0.9860 | 99.01 % | 97.67 % | 99.74 % | **98.69 %** | 0.9990 | 1.41 % |
| K-NN | Cours — paresseux | 0.9843 | 0.9871 | 99.00 % | 97.67 % | 99.69 % | **98.67 %** | 0.9968 | 1.41 % |
| Analyse discriminante linéaire | Extension — linéaire | 0.9832 | 0.9837 | 98.78 % | 97.12 % | 99.69 % | **98.39 %** | 0.9954 | 1.75 % |
| Naïve Bayes de Bernoulli | Cours — bayésien | 0.9194 | 0.9200 | 93.81 % | 87.66 % | 97.03 % | **92.11 %** | 0.9889 | 8.11 % |
| Naïve Bayes gaussien | Cours — bayésien | 0.9257 | 0.9395 | 94.07 % | 96.50 % | 87.24 % | **91.64 %** | 0.9855 | 1.88 % |
| Naïve Bayes multinomial | Cours — bayésien | 0.8964 | 0.8924 | 91.82 % | 87.36 % | 91.25 % | **89.26 %** | 0.9690 | 7.84 % |
| Zero-R (baseline) | Cours — référence | 0.0000 | 0.0000 | 62.75 % | 0.00 % | 0.00 % | **0.00 %** | 0.5000 | 0.00 % |

🏆 = modèle déployé. ★ Critère de sélection : moyenne sur **5 plis stratifiés communs à tous les modèles** (y compris vote et stacking) ; le jeu de test n'est jamais utilisé pour choisir.

### 4.2 Classification multi-classe (10 catégories) — profil complet
| Modèle | F1-macro plis communs ★ | F1-macro CV recherche | Accuracy | Acc. équilibrée | **F1-macro** | F1 pondéré | ROC-AUC OvR |
|---|---|---|---|---|---|---|---|
| **Stacking (RF+ET+XGB → LR)** 🏆 | 0.6660 | 0.6660 | 90.33 % | 77.92 % | **76.74 %** | 90.82 % | 0.9868 |
| Gradient Boosting (histogrammes) | 0.6423 | 0.6728 | 88.53 % | 71.32 % | **69.19 %** | 89.13 % | 0.9812 |
| Random Forest | 0.6642 | 0.6909 | 88.31 % | 71.68 % | **69.15 %** | 89.32 % | 0.9778 |
| Vote souple (RF+ET+XGB) | 0.6640 | 0.6640 | 88.13 % | 70.43 % | **69.08 %** | 89.17 % | 0.9786 |
| Extra Trees | 0.6530 | 0.6723 | 87.94 % | 69.62 % | **68.51 %** | 89.11 % | 0.9746 |
| Arbre de décision (CART) | 0.6136 | 0.6761 | 88.45 % | 70.53 % | **68.37 %** | 88.87 % | 0.9681 |
| XGBoost | 0.6483 | 0.6777 | 88.38 % | 68.90 % | **68.25 %** | 89.04 % | 0.9816 |
| K-NN | 0.5575 | 0.5874 | 86.19 % | 57.72 % | **58.56 %** | 85.87 % | 0.8808 |
| Régression logistique | 0.5530 | 0.5721 | 86.23 % | 57.16 % | **57.60 %** | 86.28 % | 0.9732 |
| Naïve Bayes de Bernoulli | 0.4104 | 0.4151 | 76.65 % | 53.04 % | **42.45 %** | 77.33 % | 0.9168 |
| Naïve Bayes multinomial | 0.3883 | 0.3935 | 76.61 % | 47.62 % | **39.78 %** | 76.62 % | 0.9153 |
| Naïve Bayes gaussien | 0.3374 | 0.3494 | 68.05 % | 50.48 % | **34.11 %** | 70.91 % | 0.8926 |
| Zero-R (baseline) | 0.0771 | 0.0771 | 62.75 % | 10.00 % | **7.71 %** | 48.38 % | 0.5000 |

### 4.3 Profil portable (sans `sttl`, `dttl`, `ct_state_ttl`)
| Modèle | Famille | F1 CV plis communs ★ | F1 CV recherche | Accuracy | Précision | Rappel | **F1** | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** 🏆 | Cours — ensembles (bagging) | 0.9846 | 0.9866 | 99.05 % | 97.60 % | 99.90 % | **98.74 %** | 0.9988 | 1.46 % |
| XGBoost | Extension — boosting | 0.9834 | 0.9861 | 99.01 % | 97.61 % | 99.79 % | **98.69 %** | 0.9990 | 1.45 % |
| Arbre de décision (CART) | Cours — arbres | 0.9761 | 0.9844 | 98.97 % | 97.47 % | 99.82 % | **98.63 %** | 0.9975 | 1.54 % |
| K-NN | Cours — paresseux | 0.9814 | 0.9848 | 98.86 % | 97.56 % | 99.43 % | **98.49 %** | 0.9956 | 1.48 % |
| Naïve Bayes gaussien | Cours — bayésien | 0.8895 | 0.8833 | 90.89 % | 86.17 % | 89.99 % | **88.04 %** | 0.9296 | 8.57 % |
| Zero-R (baseline) | Cours — référence | 0.0000 | 0.0000 | 62.75 % | 0.00 % | 0.00 % | **0.00 %** | 0.5000 | 0.00 % |

### 4.4 Comparaison statistique des modèles
- **Friedman** (binaire, 15 modèles × 5 plis communs) : χ² = 60.38, p = 1.01e-07 ; distance critique de Nemenyi = 9.59 rang.
- **McNemar** (svm vs voting, test) : b = 2, c = 2, p = 1.

### 4.5 Variables les plus importantes (consensus de 5 méthodes : Gini, gain, SHAP, permutation, information mutuelle)
`sttl`, `ct_state_ttl`, `dttl`, `dbytes`, `tcprtt`, `sbytes`, `byte_ratio`, `bytes_per_pkt`

### 4.6 Tests statistiques
- Normalité rejetée (Shapiro-Wilk, BH) pour **36 / 36** variables → tests non paramétriques et `log1p`.
- Mann-Whitney : **35 / 36** variables significatives (normal vs attaque) ; Kruskal-Wallis : **36 / 36** (10 catégories).

### 4.7 Non supervisé
| Algorithme | Clusters | Silhouette | Davies-Bouldin | ARI | NMI | Pureté binaire |
|---|---|---|---|---|---|---|
| K-Medoids PAM (k=6, n=1500) | 6 | 0.463 | 0.816 | 0.351 | 0.406 | 0.912 |
| K-Means++ (k=6, n=1500, même échantillon) | 6 | 0.462 | 0.813 | 0.347 | 0.403 | 0.911 |
| CLARA (k=6, 5×800) | 6 | 0.457 | 0.846 | 0.347 | 0.396 | 0.914 |
| K-Means++ (k=6, coude) | 6 | 0.453 | 0.848 | 0.343 | 0.393 | 0.913 |
| AGNES ward (k=6, n=2000) | 6 | 0.404 | 0.843 | 0.296 | 0.344 | 0.867 |
| K-Means++ (k=10 = nb catégories) | 10 | 0.511 | 0.686 | 0.272 | 0.353 | 0.896 |
| DBSCAN (eps=1.78, minPts=10) | 17 | 0.453 | 0.668 | 0.252 | 0.351 | 0.939 |
| GMM (k=11, BIC min) | 11 | 0.458 | 1.372 | 0.251 | 0.384 | 0.898 |
| AGNES complete (k=6, n=2000) | 6 | 0.308 | 1.081 | 0.174 | 0.187 | 0.720 |
| DIANA (k=6, n=700) | 6 | 0.326 | 0.897 | 0.071 | 0.119 | 0.644 |
| AGNES average (k=6, n=2000) | 6 | 0.249 | 0.754 | 0.017 | 0.043 | 0.630 |
| AGNES single (k=6, n=2000) | 6 | 0.232 | 0.580 | -0.021 | 0.014 | 0.614 |

| Détecteur d'anomalies | Apprentissage | ROC-AUC | F1 (seuil optimal) |
|---|---|---|---|
| Log-vraisemblance GMM | semi-supervisé (normal seulement) | 0.9844 | 0.9371 |
| Distance au centroïde K-Means | semi-supervisé (normal seulement) | 0.9571 | 0.8977 |
| Isolation Forest | semi-supervisé (normal seulement) | 0.8995 | 0.8364 |
| Local Outlier Factor | semi-supervisé (normal seulement) | 0.7597 | 0.6432 |
| Isolation Forest | non supervisé (train sans étiquettes) | 0.6804 | 0.6379 |

Apriori : 20527 itemsets fréquents, 48173 règles (confiance ≥ 0,7).

### 4.8 Chaîne complète PCAP → 47 features → modèle (captures avec vérité terrain)
| Capture | Profil | Flux évalués | Flux | Attaques | Rappel | Fausses alertes | F1 | Type d'attaque exact |
|---|---|---|---|---|---|---|---|---|
| Rejeu d'enregistrements réels UNSW-NB15 (TTL d'origine) | full | tous les flux | 1450 | 363 | 0.873 | 0.045 | 0.870 | 0.719 |
| Rejeu d'enregistrements réels UNSW-NB15 (TTL d'origine) | full | jamais vus à l'entraînement | 1010 | 85 | 1.000 | 0.050 | 0.787 | 0.953 |
| Rejeu d'enregistrements réels UNSW-NB15 (TTL d'origine) | portable | tous les flux | 1450 | 363 | 1.000 | 0.046 | 0.936 | 0.730 |
| Rejeu d'enregistrements réels UNSW-NB15 (TTL d'origine) | portable | jamais vus à l'entraînement | 1010 | 85 | 1.000 | 0.051 | 0.783 | 0.929 |
| Même rejeu, TTL d'hôtes modernes (64) | full | tous les flux | 1450 | 363 | 0.003 | 0.000 | 0.005 | 0.000 |
| Même rejeu, TTL d'hôtes modernes (64) | full | jamais vus à l'entraînement | 1010 | 85 | 0.012 | 0.000 | 0.023 | 0.000 |
| Même rejeu, TTL d'hôtes modernes (64) | portable | tous les flux | 1450 | 363 | 1.000 | 0.046 | 0.936 | 0.730 |
| Même rejeu, TTL d'hôtes modernes (64) | portable | jamais vus à l'entraînement | 1010 | 85 | 1.000 | 0.051 | 0.783 | 0.929 |
| Scénarios synthétiques (topologie du banc de test) | full | tous les flux | 428 | 274 | 0.755 | 0.000 | 0.861 | 0.288 |
| Scénarios synthétiques (topologie du banc de test) | portable | tous les flux | 428 | 274 | 0.347 | 0.097 | 0.495 | 0.000 |
| Scénarios synthétiques (TTL 64/128) | full | tous les flux | 316 | 185 | 0.000 | 0.000 | 0.000 | 0.000 |
| Scénarios synthétiques (TTL 64/128) | portable | tous les flux | 316 | 185 | 0.351 | 0.160 | 0.480 | 0.000 |

**Artefact TTL** : sur le même trafic réel rejoué avec des TTL de 64, le profil complet ne détecte presque plus rien, alors que le profil portable garde les mêmes performances → profil portable recommandé pour une capture d'un autre réseau.

Test aller-retour (enregistrement réel → PCAP → features) : `state` 99.8 %, `service` 95.0 %, `sbytes` 99.2 %, `dbytes` 95.3 %, `spkts` 100.0 %, `dpkts` 100.0 %, `sttl` 100.0 %, `dttl` 100.0 %, `dur` 100.0 %, `smeansz` 99.6 %, `dmeansz` 95.3 %, `sload` 98.7 %, `dload` 95.3 %, `synack` 94.2 %, `ackdat` 94.3 %, `tcprtt` 94.0 %, `trans_depth` 100.0 %, `res_bdy_len` 100.0 %, `ct_flw_http_mthd` 100.0 %, `is_ftp_login` 100.0 %, `ct_state_ttl` 100.0 %, `is_sm_ips_ports` 100.0 %

Fidélité des formules de l'extracteur sur le CSV officiel : `sload` 99.23 %, `dload` 99.86 %, `smeansz` 100.00 %, `dmeansz` 100.00 %, `tcprtt` 100.00 %, `ct_state_ttl` 99.98 %, `is_sm_ips_ports` 99.97 %, `swin (TCP)` 99.99 %
<!-- RESULTS:END -->

### Justification des choix d'optimisation
- **Validation croisée stratifiée 5-fold** (K ≥ 5 exigé) : chaque pli conserve la proportion des 10 catégories ; le
  préprocesseur est *dans* le pipeline, donc ré-ajusté sur chaque pli (aucune fuite).
- **GridSearchCV** pour les espaces discrets et petits (K-NN : K, distance, pondération ; CART : critère Gini/entropie,
  profondeur, élagage ccp_α ; Naïve Bayes : lissage de Laplace α / var_smoothing ; SVM : noyaux linéaire/RBF/polynomial, C, γ ;
  régression logistique : C, class_weight ; LDA ; AdaBoost) ; **RandomizedSearchCV** pour les espaces continus des
  ensembles (Random Forest, Extra Trees, HistGB, XGBoost) — meilleure couverture à budget égal (Bergstra & Bengio 2012).
- **Scoring F1** (binaire) / **F1-macro** (multi-classe) : robustes au déséquilibre, contrairement à l'accuracy.
- **Sélection sur la CV, jamais sur le test**, puis comparaison statistique (Friedman + Nemenyi, Wilcoxon-Holm, McNemar).
- K-NN et SVM (complexité O(n²)) : recherche sur un sous-échantillon stratifié (20 000 / 8 000 flux), documenté.

### Justification des tests statistiques
| Test | Question | Pourquoi ce test |
|---|---|---|
| Shapiro-Wilk, D'Agostino K² | Les variables sont-elles normales ? | Conditionne le choix paramétrique / non paramétrique ; SW est le plus puissant pour n ≤ 5 000 |
| Mann-Whitney U (+ r rang-bisériale) | Une variable diffère-t-elle entre normal et attaque ? | Non paramétrique (normalité rejetée), 2 groupes indépendants ; la taille d'effet compense n très grand |
| Kruskal-Wallis (+ ε²) | … entre les 10 catégories ? | Généralisation non paramétrique de l'ANOVA |
| χ² d'indépendance (+ V de Cramér) | proto / state / service liés à la cible ? | Variables catégorielles ; modalités rares exclues (effectifs théoriques ≥ 5) |
| Spearman | Corrélations / redondance ? | Relations monotones, robuste aux queues lourdes |
| Benjamini-Hochberg | Contrôle des faux positifs | Des dizaines de tests simultanés |
| Friedman, Nemenyi, Wilcoxon, McNemar | Les écarts entre modèles sont-ils réels ? | Comparaison sur plis communs / mêmes exemples de test |

---

## 5. Architecture du dépôt
```
Network-Intrusion-Detection-System/
├── README.md · requirements.txt · app.py (lanceur) · pytest.ini
├── data/
│   ├── raw/              # CSV UNSW-NB15 d'origine, lecture seule (externe, non versionné)
│   ├── interim/          # échantillon stratifié brut + jeu nettoyé
│   ├── processed/        # train.csv / test.csv + split_metadata.json
│   └── pcaps/            # captures de démonstration + vérité terrain par flux
├── notebooks/            # 01_eda_and_cleaning → 07_extraction_pcap (+ build_notebooks.py)
├── src/
│   ├── app.py            # application Flask (point d'entrée, routage)
│   ├── services.py       # logique de l'application (EDA filtrable, prédiction, PCAP)
│   ├── preprocessing.py  # chargement, nettoyage, pipeline de prétraitement
│   ├── models.py         # catalogue des modèles, chargement, prédiction
│   ├── features.py       # schéma des 47 features (source unique de vérité)
│   ├── config.py         # chemins et constantes
│   ├── train.py          # recherches d'hyperparamètres, évaluation, déploiement
│   ├── evaluate.py       # métriques, courbes ROC/PR, seuils
│   ├── predict.py        # point d'entrée de prédiction (CLI)
│   ├── prepare_data.py · run_analysis.py · run_unsupervised.py
│   ├── pcap/             # reader, flows (Argus), app_layer (Bro), writer, generator
│   ├── evaluation/       # statistical_tests, explain (SHAP)
│   ├── unsupervised/     # clustering (PAM, CLARA, DIANA…), association (Apriori)
│   ├── templates/ · static/   # interface web
│   └── visualization.py  # figures statiques
├── models/
│   ├── binaire/          # pipeline.joblib + metadata.json (détecteur déployé)
│   ├── multiclasse/      # pipeline.joblib + metadata.json (catégorie déployée)
│   └── all/              # les ~15 modèles comparés + registry.json + modèles non supervisés
├── reports/
│   ├── metrics/          # tableaux CSV, curves.json, analysis.json, unsupervised.json
│   └── figures/          # eda/ et evaluation/
├── docs/                 # Rapport_MiniProject.pdf (+ source HTML dans docs/rapport/)
├── scripts/              # download_data.py, train.py, predict_pcap.py, update_readme_results.py, build_report.py
└── tests/                # pytest
```

---

## 6. Application web

`python app.py` → http://127.0.0.1:5000 (fonctionne hors-ligne : Plotly.js est servi depuis le paquet Python).

| Page | Contenu |
|---|---|
| **Accueil** | KPIs, pipeline, composition des données, meilleurs modèles |
| **Dashboard EDA** | ① statistiques descriptives ② valeurs manquantes (brut vs nettoyé) ③ histogrammes + KDE ④ fréquences catégorielles ⑤ heatmap de corrélation (Spearman/Pearson) ⑥ **filtres temps réel** (catégories, classe, protocole, service, état, plage numérique) + boîtes à moustaches, nuage de points |
| **Tests statistiques** | normalité, Mann-Whitney, Kruskal-Wallis, χ², Spearman, outliers — avec justification |
| **Modèles & résultats** | tableau multicritère, ROC, PR, matrices de confusion, résultats de CV, seuil, courbes d'apprentissage, Friedman/Nemenyi/McNemar ; binaire / multi-classe ; profil complet / portable |
| **Explicabilité** | 5 méthodes d'importance, concordance, SHAP par catégorie, ablation |
| **Prédiction temps réel** | formulaire des 39 features avec exemples réels, mise à jour instantanée : probabilité, catégorie, explication SHAP et textuelle, cluster, score d'anomalie |
| **Analyse PCAP** | téléversement pcap/pcapng/csv, extraction des 47 features, alertes, chronologie, IP suspectes, détail d'un flux, export CSV, évaluation si vérité terrain |
| **Non supervisé** | coude, silhouette, comparaison des 7 algorithmes, ACP, dendrogrammes, DBSCAN, GMM, anomalies |
| **Règles d'association** | règles Apriori filtrables (support, confiance, lift) |
| **Génération des tuples** | cadre de la thèse, fidélité des formules, dictionnaire des 47 features |

---

## 7. Limites et perspectives
- **Artefact TTL** : dans le banc de test, les attaquants émettent en TTL 254/62 ; `sttl` devient la variable la plus
  discriminante. D'où un **profil portable** (sans `sttl`, `dttl`, `ct_state_ttl`) recommandé pour d'autres réseaux.
- **Bruit d'étiquetage** : 8 053 vecteurs identiques portent des étiquettes différentes (surtout DoS / Exploits /
  Analysis / Backdoor) → plafond du F1-macro.
- **Features `ct_*`** : reproduites approximativement (le CSV publié est un sous-ensemble des enregistrements Argus).
- **Captures de démonstration synthétiques** ; perspective : valider sur les PCAP originaux (100 Go) et sur CIC-UNSW-NB15.
- Perspectives : apprentissage incrémental, calibration des probabilités, capture en direct.

### Références
- N. Moustafa, J. Slay. *UNSW-NB15: a comprehensive data set for network intrusion detection systems*. MilCIS, 2015.
- N. Moustafa. *Designing an online and reliable statistical anomaly detection framework for dealing with large high-speed network traffic*. Thèse, UNSW, 2017.
- S. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS, 2017.
- L. Kaufman, P. Rousseeuw. *Finding Groups in Data* (PAM, CLARA, DIANA). Wiley, 1990.
- R. Agrawal, R. Srikant. *Fast Algorithms for Mining Association Rules*. VLDB, 1994.
