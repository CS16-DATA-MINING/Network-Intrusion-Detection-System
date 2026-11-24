"""Génère les notebooks du projet (nbformat). Usage : python notebooks/build_notebooks.py

Les notebooks appellent le code de ``src/`` (source unique de vérité) et
expliquent, étape par étape, le rôle de chaque traitement.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """# Configuration : le notebook s'exécute depuis notebooks/, on ajoute la racine du dépôt au chemin
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from src import config as C
from src import visualization as V
pd.set_option('display.max_columns', 60); pd.set_option('display.width', 200)
print('Racine du projet :', ROOT.name)"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    nbf.write(nb, HERE / name)
    print("écrit", name)


# =============================================================================
# 01 — EDA & nettoyage
# =============================================================================
nb01 = [
    md("""
# 01 — Compréhension des données, nettoyage et analyse exploratoire (EDA)

**Projet :** système de détection d'intrusions réseau (NIDS) sur le jeu **UNSW-NB15**.

| Élément | Valeur |
|---|---|
| Source | UNSW Canberra — Moustafa & Slay (2015), thèse de N. Moustafa (2017) |
| Volume brut | 4 fichiers CSV, **2 540 047 flux**, **49 colonnes** |
| Cibles | `label` (0 normal / 1 attaque) et `attack_cat` (10 catégories) |

**Rôle de ce notebook :** comprendre la structure des données, mesurer leur qualité (valeurs manquantes,
codages incohérents, doublons), les nettoyer, puis explorer les distributions et vérifier par des
**tests statistiques** quelles variables distinguent le trafic normal des attaques. Ces constats
justifient les choix de prétraitement du notebook 02.
"""),
    code(SETUP),
    md("""
## 1. Chargement et échantillonnage stratifié

**Pourquoi échantillonner ?** Le jeu complet (2,54 M lignes, 87 % de trafic normal) est trop lourd pour
comparer 15 modèles avec validation croisée, et la classe rare *Worms* ne compte que 174 flux.
On construit un **échantillon stratifié reproductible** (`random_state=42`) : 60 000 flux normaux tirés
uniformément et **jusqu'à 8 000 flux par catégorie d'attaque** — toutes les lignes des classes rares sont gardées.

Le code est dans `src/preprocessing.py` (`load_raw_sample`) et `src/prepare_data.py`. S'il a déjà été
exécuté, on relit simplement les fichiers produits.
"""),
    code("""
from src import prepare_data
if not C.CLEAN_FILE.exists():
    prepare_data.main()          # lecture des 4 CSV bruts (~15 s), échantillonnage, nettoyage, split
meta = json.loads(C.SPLIT_META_FILE.read_text(encoding='utf-8'))
rs = meta['raw_stats']
print(f"Lignes par fichier : {rs['files']}")
print(f"Total brut : {rs['full_total']:,} flux  |  échantillon : {rs['sample_total']:,}")
full = pd.Series(rs['full_counts']).reindex(C.ATTACK_CATEGORIES)
sample = pd.Series(rs['sample_counts']).reindex(C.ATTACK_CATEGORIES)
display(pd.DataFrame({'jeu complet': full, '% complet': (100 * full / full.sum()).round(3),
                      'échantillon': sample, '% échantillon': (100 * sample / sample.sum()).round(2)}))
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
V.barh(axes[0], full.index, full.values, color=V.class_colors(full.index), fmt='{:,.0f}')
axes[0].set_xscale('log'); axes[0].set_title('Jeu complet (échelle log)')
V.barh(axes[1], sample.index, sample.values, color=V.class_colors(sample.index), fmt='{:,.0f}')
axes[1].set_title('Échantillon stratifié')
fig.suptitle('Distribution des catégories — vert : normal, rouge : attaques'); plt.tight_layout()
V.save(fig, '01_class_distribution'); plt.show()
"""),
    md("""
## 2. Qualité des données brutes

On inspecte l'échantillon **avant** nettoyage. Points connus de UNSW-NB15 que l'on vérifie :
- `attack_cat` vide pour le trafic normal ;
- `ct_flw_http_mthd`, `is_ftp_login` vides dans les fichiers 2 à 4 (flux non HTTP/FTP) ;
- `ct_ftp_cmd` contenant la chaîne `' '` ;
- ports source/destination parfois **hexadécimaux** (`0x000b`) ou `'-'` ;
- espaces parasites (`' Fuzzers'`) et doubles libellés (`Backdoor` / `Backdoors`).
"""),
    code("""
raw = pd.read_csv(C.SAMPLE_FILE, low_memory=False)
blank = raw.apply(lambda s: s.isna() | s.astype(str).str.strip().eq(''))
miss = (blank.mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]
display(miss.round(2).to_frame('% manquant (brut)'))
fig, ax = plt.subplots(figsize=(7, 2.8))
V.barh(ax, miss.index, miss.values, color=V.SERIES[1], fmt='{:.1f} %')
ax.set_xlabel('% de valeurs manquantes'); ax.set_title('Valeurs manquantes — données brutes')
V.save(fig, '01_missing_values'); plt.show()
hexa = raw[['sport', 'dsport']].astype(str).apply(lambda s: s.str.startswith('0x').sum())
print('Ports hexadécimaux :', hexa.to_dict(), '| libellés attack_cat bruts :', raw['attack_cat'].dropna().unique()[:12])
"""),
    md("""
## 3. Nettoyage

Fonction `clean_dataframe` (identique pour le CSV et pour les flux extraits d'un PCAP — voir notebook 07) :

| Problème | Traitement | Justification |
|---|---|---|
| `attack_cat` vide | → `Normal` | la ligne est normale (`label = 0`) |
| `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd` vides | → 0 | absence d'événement HTTP/FTP, pas une donnée inconnue |
| `is_ftp_login` > 1 | → 1 | variable binaire par définition (Alg. 3.5) |
| ports hexadécimaux / `-` | conversion entière / NaN | identifiants, non utilisés comme features |
| catégories avec espaces / casse | normalisation | éviter des modalités fantômes |
| doublons exacts (features + cible) | suppression | éviter un biais d'apprentissage et une fuite train/test |

Les **identifiants** (`srcip`, `sport`, `dstip`, `dsport`, `stime`, `ltime`) et les **numéros de séquence
initiaux** (`stcpb`, `dtcpb`, aléatoires par construction) sont exclus des features : un modèle qui apprend
les adresses du banc de test ne généralise pas.
"""),
    code("""
rep = meta['cleaning_report']
display(pd.Series({k: v for k, v in rep.items() if not isinstance(v, dict)}, name='valeur').to_frame())
df = pd.read_csv(C.CLEAN_FILE, low_memory=False)
print(f"Jeu nettoyé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes | features modèle : {len(C.FEATURES)}")
print('Valeurs manquantes restantes dans les features :', int(df[C.FEATURES].isna().sum().sum()))
print(f"Vecteurs identiques étiquetés différemment : {rep['conflicting_vectors']:,} "
      "(bruit d'étiquetage connu de UNSW-NB15 — borne la performance multi-classe)")
"""),
    md("## 4. Statistiques descriptives\nTendance centrale (moyenne, médiane), dispersion (écart-type, IQR) et forme (asymétrie, aplatissement)."),
    code("""
desc = df[C.NUMERIC_FEATURES].describe().T
desc['asymétrie'] = df[C.NUMERIC_FEATURES].skew(); desc['aplatissement'] = df[C.NUMERIC_FEATURES].kurt()
display(desc.round(3))
display(df[C.CATEGORICAL_FEATURES + ['attack_cat']].describe().T)
"""),
    md("""
**Lecture :** la plupart des variables de volume (`sbytes`, `dbytes`, `sload`, `dload`, `sjit`…) ont une
asymétrie > 10 et une moyenne très supérieure à la médiane : distributions à **queue lourde**. On travaillera
en échelle `log1p` pour les visualiser et les modéliser.
"""),
    md("## 5. Distributions des variables numériques (histogrammes + KDE)"),
    code("""
key = ['sbytes', 'dbytes', 'dur', 'sttl', 'smeansz', 'ct_srv_src', 'sload', 'dmeansz']
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, c in zip(axes.ravel(), key):
    for lab, col, name in ((0, V.NORMAL, 'Normal'), (1, V.ATTACK, 'Attaque')):
        x = np.log1p(df.loc[df.label == lab, c].clip(lower=0))
        sns.histplot(x, bins=40, stat='density', color=col, alpha=0.35, ax=ax, label=name, edgecolor=None)
        if x.nunique() > 2:
            sns.kdeplot(x, color=col, ax=ax, linewidth=1.8, warn_singular=False)
    ax.set_title(f'log1p({c})'); ax.set_xlabel(''); ax.set_ylabel('')
axes[0, 0].legend()
fig.suptitle('Distributions par classe (échelle log1p) — histogrammes et densités KDE'); plt.tight_layout()
V.save(fig, '01_numeric_distributions'); plt.show()
"""),
    md("""
**Observations :**
- `sttl` est quasi binaire : 31 (normal) vs 254/62 (attaques). C'est la variable la plus séparatrice, mais
  c'est en partie un **artefact de la topologie du banc de test** (thèse §3.3) — nous évaluerons un profil
  de modèle « portable » sans TTL.
- Les attaques ont des `dbytes` nuls bien plus souvent (flux unidirectionnels, état `INT`).
- `ct_srv_src` (connexions répétées vers le même service) est élevé pour *Generic* (rafales DNS).
"""),
    md("## 6. Fréquences des variables catégorielles"),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, C.CATEGORICAL_FEATURES):
    top = df[c].value_counts().index[:10]
    ct = pd.crosstab(df[c], df['label']).reindex(top)
    ax.bar(ct.index.astype(str), ct[0], color=V.NORMAL, label='Normal')
    ax.bar(ct.index.astype(str), ct[1], bottom=ct[0], color=V.ATTACK, label='Attaque')
    ax.set_title(c); ax.tick_params(axis='x', rotation=45)
axes[0].legend(); plt.tight_layout(); V.save(fig, '01_categorical_frequencies'); plt.show()
display(pd.crosstab(df['attack_cat'], df['proto']).loc[:, lambda t: t.sum().nlargest(6).index])
"""),
    md("## 7. Boîtes à moustaches par catégorie d'attaque"),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, c in zip(axes, ['sbytes', 'dur', 'smeansz']):
    d = df.assign(v=np.log1p(df[c].clip(lower=0)))
    order = C.ATTACK_CATEGORIES
    sns.boxplot(data=d, x='attack_cat', y='v', order=order, ax=ax, color=V.SERIES[0], fliersize=1, linewidth=1)
    ax.set_title(f'log1p({c}) par catégorie'); ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); V.save(fig, '01_boxplots_by_category'); plt.show()
"""),
    md("## 8. Corrélations (Spearman)\nSpearman (sur les rangs) est préféré à Pearson : il capte les relations monotones et résiste aux queues lourdes."),
    code("""
cols = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload', 'spkts', 'dpkts', 'smeansz',
        'dmeansz', 'sjit', 'djit', 'tcprtt', 'synack', 'ackdat', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
        'ct_src_ltm', 'ct_dst_src_ltm', 'ct_state_ttl', 'label']
corr = df[cols].corr(method='spearman')
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap=V.DIV_CMAP, vmin=-1, vmax=1, ax=ax, linewidths=0.4, linecolor='#fcfcfb', cbar_kws={'shrink': 0.7})
ax.set_title('Matrice de corrélation de Spearman'); V.save(fig, '01_spearman_heatmap'); plt.show()
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=np.abs, ascending=False)
display(pairs.head(12).round(3).to_frame('ρ'))
"""),
    md("""
**Lecture :** blocs très corrélés (`sbytes`–`spkts`–`sloss`, `dbytes`–`dpkts`–`dloss`, `synack`–`ackdat`–`tcprtt`,
famille `ct_*`). La redondance n'est pas grave pour les arbres mais déstabilise les modèles linéaires et
Naïve Bayes (hypothèse d'indépendance) : le pipeline retire une variable de chaque paire |r| > 0,95 (ajusté sur le train).
"""),
    md("""
## 9. Tests statistiques

Six familles de tests, chacune **justifiée** par la nature des variables (code : `src/evaluation/statistical_tests.py`).
Comme on réalise plusieurs dizaines de tests, les p-values sont corrigées par **Benjamini-Hochberg** (FDR 5 %) :
sans correction, on attendrait ≈ 5 % de faux positifs parmi les tests.

Les tests sont menés sur le **train uniquement** (le test reste intact pour l'évaluation finale).
"""),
    code("""
from src.evaluation import statistical_tests as ST
train = pd.read_csv(C.TRAIN_FILE, low_memory=False)
num = C.NUMERIC_FEATURES
"""),
    md("""
### 9.1 Normalité — Shapiro-Wilk et D'Agostino K²
*H₀ : la variable suit une loi normale.* Shapiro-Wilk est le test le plus puissant pour n ≤ 5 000 (on tire
4 000 observations : au-delà, le test rejette pour des écarts minimes). D'Agostino K² combine asymétrie et
aplatissement et confirme le résultat.
"""),
    code("""
norm = ST.normality_tests(train, num)
display(norm[['feature', 'shapiro_W', 'shapiro_p_bh', 'dagostino_K2', 'skewness', 'normal_at_5pct']].round(4).head(15))
print(f"Normalité rejetée (p_BH < 0,05) pour {(~norm.normal_at_5pct).sum()} / {norm.shapiro_p.notna().sum()} variables testées.")
"""),
    md("**Conséquence :** aucune variable n'est normale → on utilise des tests **non paramétriques** (Mann-Whitney, Kruskal-Wallis, Spearman) et une transformation `log1p` + standardisation dans le prétraitement."),
    md("""
### 9.2 Mann-Whitney U — numérique vs classe binaire
*H₀ : les distributions normal / attaque sont identiques.* Alternative non paramétrique au test t (deux groupes
indépendants, variables non normales). Avec n ≈ 73 000, presque tout est significatif : on rapporte donc la
**taille d'effet** r rang-bisériale (|r| > 0,5 : effet fort).
"""),
    code("""
mw = ST.mann_whitney(train, num)
display(mw[['feature', 'median_normal', 'median_attack', 'rank_biserial_r', 'effect_size', 'p_bh']].head(15).round(4))
fig, ax = plt.subplots(figsize=(8, 7))
top = mw.head(25)
ax.barh(top.feature, top.rank_biserial_r, color=[V.SERIES[0] if r > 0 else V.SERIES[1] for r in top.rank_biserial_r])
ax.invert_yaxis(); ax.axvline(0, color='#898781', lw=0.8)
ax.set_xlabel('r rang-bisériale (> 0 : valeurs plus élevées chez les attaques)'); ax.set_title('Mann-Whitney U — taille d\\'effet')
V.save(fig, '01_mann_whitney_effects'); plt.show()
print(f"{mw.significant.sum()} / {len(mw)} variables significatives après correction BH ; effet fort : {(mw.abs_effect > 0.5).sum()}")
"""),
    md("""
### 9.3 Kruskal-Wallis — numérique vs 10 catégories
Généralisation non paramétrique de l'ANOVA à un facteur. Taille d'effet ε² = (H − k + 1)/(n − k).
"""),
    code("""
kw = ST.kruskal_wallis(train, num)
display(kw.head(12).round(4))
"""),
    md("""
### 9.4 Chi² d'indépendance + V de Cramér — catégoriel vs cible
*H₀ : la variable est indépendante de la cible.* Condition de validité : effectifs théoriques ≥ 5 → les modalités
trop rares pour l'atteindre (compte tenu de la plus petite classe, ex. Worms) sont regroupées dans « autre »
(colonne `pct_expected_lt5` = 0 % pour tous les tests). Le V de Cramér (0 à 1) mesure la force de l'association indépendamment de n.
"""),
    code("""
chi = pd.concat([ST.chi2_tests(train, C.CATEGORICAL_FEATURES, 'label'), ST.chi2_tests(train, C.CATEGORICAL_FEATURES, 'attack_cat')])
display(chi.round(4))
"""),
    md("### 9.5 Corrélation de Spearman avec la cible"),
    code("""
sp = ST.spearman_with_target(train, num)
display(sp.head(12).round(4))
"""),
    md("""
### 9.6 Valeurs aberrantes (règle de Tukey, 1,5 × IQR) — diagnostic
On mesure la proportion de valeurs hors de [Q1 − 1,5 IQR ; Q3 + 1,5 IQR] et, parmi elles, la part d'attaques.
"""),
    code("""
out = ST.outlier_summary(train, num)
display(out.head(12).round(2))
"""),
    md("""
**Décision :** les « outliers » sont majoritairement des **attaques** (ce que l'on cherche à détecter !) :
on ne les supprime pas. On réduit leur influence par `log1p` + Z-score et l'usage de modèles robustes (arbres).

## 10. Conclusions de l'EDA → choix de prétraitement

1. **Non-normalité généralisée** (Shapiro-Wilk) → tests non paramétriques, `log1p` des variables asymétriques, standardisation.
2. **Variables discriminantes** (Mann-Whitney, Kruskal-Wallis, χ²) : `sttl`, `ct_state_ttl`, `dttl`, `dmeansz`,
   `dbytes`, `state`, `proto`, `service`, `ct_srv_src`… → toutes conservées ; le notebook 06 confirmera
   par l'importance des variables.
3. **Redondance forte** (Spearman) → filtre de corrélation |r| > 0,95 ajusté sur le train.
4. **Déséquilibre** des catégories → split stratifié, métriques F1 / F1-macro et rappel, `class_weight` testé.
5. **Artefact TTL** → profil de modèle « portable » sans `sttl`, `dttl`, `ct_state_ttl`.
"""),
]

# =============================================================================
# 02 — Prétraitement & encodage
# =============================================================================
nb02 = [
    md("""
# 02 — Prétraitement et encodage

**Rôle :** transformer les flux nettoyés en une matrice numérique adaptée à chaque famille de modèles,
**sans fuite d'information** : toutes les transformations sont *ajustées uniquement sur le train* puis
appliquées telles quelles au test et aux flux extraits d'un PCAP.

Le pipeline (`src/preprocessing.py`, `build_preprocessor`) est enchaîné au classifieur dans un
`Pipeline` scikit-learn : pendant la validation croisée, il est ré-ajusté dans chaque pli.
"""),
    code(SETUP),
    md("## 1. Split stratifié train / test (80 / 20)\nLa stratification sur les 10 catégories garantit que les classes rares (Worms : 171 flux) sont présentes dans les deux jeux dans la même proportion."),
    code("""
train = pd.read_csv(C.TRAIN_FILE, low_memory=False); test = pd.read_csv(C.TEST_FILE, low_memory=False)
split = pd.DataFrame({'train': train.attack_cat.value_counts(), 'test': test.attack_cat.value_counts()}).reindex(C.ATTACK_CATEGORIES)
split['% train'] = (100 * split.train / split.train.sum()).round(2); split['% test'] = (100 * split.test / split.test.sum()).round(2)
display(split)
print(f"train : {len(train):,} | test : {len(test):,} | taux d'attaque train {train.label.mean():.3f} / test {test.label.mean():.3f}")
"""),
    md("""
## 2. Les étapes du pipeline

| # | Étape | Rôle |
|---|---|---|
| 1 | `FeatureEngineer` | ajoute 5 features dérivées : `total_bytes`, `total_pkts`, `byte_ratio`, `pkt_ratio`, `bytes_per_pkt` (calculables aussi depuis un PCAP) |
| 2 | `SimpleImputer(median)` | ne jamais planter en inférence si une valeur manque |
| 3 | `SkewLog1p(seuil=2)` | `log1p` sur les colonnes dont l'asymétrie **du train** dépasse 2 (choix guidé par les tests de normalité) |
| 4 | `CorrelationFilter(0,95)` | retire une variable de chaque paire |r| > 0,95 (redondance) |
| 5 | `StandardScaler` (Z-score) | même échelle pour K-NN, SVM, régression logistique ; `MinMaxScaler` pour MultinomialNB (entrées positives) |
| 6 | `OneHotEncoder` | `proto`, `state`, `service` ; modalités < 30 occurrences regroupées (`infrequent`), modalités inconnues tolérées (indispensable pour un PCAP réel) |
"""),
    code("""
from src.preprocessing import build_preprocessor, feature_frame
Xtr, Xte = feature_frame(train), feature_frame(test)
pre = build_preprocessor().fit(Xtr)            # ajusté sur le TRAIN uniquement
Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
names = list(pre.get_feature_names_out())
ct = pre.named_steps['columns']; num_pipe = ct.named_transformers_['num']; num_cols = ct.transformers_[0][2]
logged = [c for c, m in zip(num_cols, num_pipe.named_steps['log'].log_mask_) if m]
dropped = [c for c, m in zip(num_cols, num_pipe.named_steps['corr'].keep_mask_) if not m]
print(f"Entrée : {Xtr.shape[1]} colonnes → sortie : {Ztr.shape[1]} colonnes")
print(f"log1p appliqué à {len(logged)} variables : {logged}")
print(f"Supprimées par le filtre de corrélation : {dropped}")
onehot = ct.named_transformers_['cat'].named_steps['onehot']
for c, cats in zip(C.CATEGORICAL_FEATURES, onehot.categories_):
    print(f"  {c}: {len(cats)} modalités vues au train, {len([n for n in names if n.startswith(c + '_')])} colonnes One-Hot")
"""),
    md("## 3. Effet des transformations sur une variable à queue lourde"),
    code("""
i = names.index('sbytes')
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].hist(train.sbytes, bins=60, color=V.SERIES[0]); axes[0].set_title('sbytes brut (asymétrie %.0f)' % train.sbytes.skew())
axes[1].hist(np.log1p(train.sbytes), bins=60, color=V.SERIES[0]); axes[1].set_title('log1p(sbytes)')
axes[2].hist(Ztr[:, i], bins=60, color=V.SERIES[2]); axes[2].set_title('après log1p + Z-score (μ≈0, σ≈1)')
plt.tight_layout(); V.save(fig, '02_transformations_sbytes'); plt.show()
print('moyenne/écart-type après standardisation (train) :', Ztr[:, i].mean().round(4), Ztr[:, i].std().round(4))
print('moyenne/écart-type sur le test (non ré-ajusté)  :', Zte[:, i].mean().round(4), Zte[:, i].std().round(4))
"""),
    md("""
Le test a une moyenne proche mais pas exactement nulle : c'est **normal et souhaité** — les paramètres
(moyenne, écart-type, seuils, catégories) viennent du train seul ; les recalculer sur le test serait une fuite.

## 4. Profil « portable » (sans TTL)
Même pipeline en retirant `sttl`, `dttl`, `ct_state_ttl` (artefact du banc de test).
"""),
    code("""
pre_p = build_preprocessor(drop=C.TTL_FEATURES).fit(Xtr)
print('Colonnes profil portable :', len(pre_p.get_feature_names_out()), '| profil complet :', len(names))
"""),
    md("## 5. Compatibilité avec les flux extraits d'un PCAP\nLe même pipeline accepte les tuples produits par l'extracteur (notebook 07) — y compris des protocoles ou services jamais vus."),
    code("""
from src.pcap import pcap_to_flows
from src.preprocessing import clean_dataframe
pcap = C.SAMPLE_PCAP_DIR / 'demo_unsw_testbed.pcap'
if not pcap.exists():
    from src.pcap.generator import generate_demo_pcap; generate_demo_pcap(pcap)
flows = pcap_to_flows(pcap)
Zp = pre.transform(feature_frame(clean_dataframe(flows)))
print(f"{len(flows)} flux extraits → matrice {Zp.shape}, valeurs finies : {np.isfinite(Zp).all()}")
"""),
]

# =============================================================================
# 03 — Modèles
# =============================================================================
nb03 = [
    md("""
# 03 — Entraînement, optimisation et évaluation des modèles

**Rôle :** comparer les algorithmes du cours (Zero-R, K-NN, arbre de décision, Random Forest, Naïve Bayes,
SVM) et des extensions classiques (régression logistique, LDA, Extra Trees, AdaBoost, Gradient Boosting,
XGBoost, vote souple, stacking) sur deux tâches :
- **binaire** : normal vs attaque (priorité au **rappel** : une attaque manquée coûte plus cher qu'une fausse alerte) ;
- **multi-classe** : 10 catégories.

**Protocole :**
1. chaque modèle = `Pipeline(prétraitement → classifieur)` ;
2. `GridSearchCV` (grilles discrètes) ou `RandomizedSearchCV` (espaces continus) avec **`StratifiedKFold(5)`**,
   scoring F1 (binaire) / F1-macro (multi-classe), métriques secondaires enregistrées ;
3. ré-ajustement sur tout le train, évaluation **une seule fois** sur le test ;
4. le modèle déployé est le meilleur en **moyenne sur 5 plis communs à tous les modèles** (y compris vote et stacking) :
   comparaison équitable, sans utiliser le test (choisir sur le test biaiserait l'estimation).

**L'entraînement est réalisé dans ce notebook** : les cellules ci-dessous appellent `src/train.py`
(le même code est exécutable sans Jupyter par `python scripts/train.py`). Budgets de recherche réduits pour
tenir sur une machine de 8 Go de RAM (grilles resserrées, 8 à 10 tirages aléatoires pour les ensembles,
réglage de K-NN et SVM sur sous-échantillon) — documentés dans `src/models.py`. Les résultats sont
sauvegardés après chaque tâche : si l'exécution est interrompue, la relance reprend où elle s'était arrêtée
(`RESUME = True`).
"""),
    code(SETUP),
    code("""
from src import train as TE
RESUME = True                      # False : tout ré-entraîner depuis zéro
reg, curves = TE.start(resume=RESUME)
train, test = TE.load_split()
print(f"train : {len(train):,} flux | test : {len(test):,} flux | parallélisme NIDS_JOBS = {TE.N_JOBS}")
"""),
    md("""
## 1. Démonstration du mécanisme d'optimisation (exemple K-NN)

Avant de lire les résultats complets, on illustre `GridSearchCV` sur un petit échantillon : étude du
paramètre **K**, de la **distance** (Manhattan p=1, Euclidienne p=2) et de la pondération, avec standardisation.
"""),
    code("""
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing import build_preprocessor, feature_frame
train = pd.read_csv(C.TRAIN_FILE, low_memory=False); test = pd.read_csv(C.TEST_FILE, low_memory=False)
demo = train.sample(4000, random_state=0)
pipe = Pipeline([('prep', build_preprocessor()), ('clf', KNeighborsClassifier())])
grid = {'clf__n_neighbors': [1, 3, 5, 9, 15, 25], 'clf__p': [1, 2]}
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1', n_jobs=2)
gs.fit(feature_frame(demo), demo.label)
res = pd.DataFrame(gs.cv_results_)[['param_clf__n_neighbors', 'param_clf__p', 'mean_test_score', 'std_test_score']]
display(res.pivot(index='param_clf__n_neighbors', columns='param_clf__p', values='mean_test_score').round(4))
print('Meilleur :', gs.best_params_, 'F1 CV =', round(gs.best_score_, 4))
"""),
    md("""
## 2. Entraînement de tous les modèles

Pour chaque (tâche, profil) : recherche d'hyperparamètres en CV 5 plis pour chaque modèle, ré-ajustement sur
tout le train, évaluation sur le test, scores par pli sur des plis communs (pour les tests de Friedman/Wilcoxon),
sauvegarde du pipeline dans `models/all/`. Les vote souple et stacking combinent les meilleurs RF, Extra Trees et XGBoost.
"""),
    code("t_bin = TE.train_task('binary', 'full', reg, curves, train, test, skip_done=RESUME)"),
    code("t_multi = TE.train_task('multiclass', 'full', reg, curves, train, test, skip_done=RESUME)"),
    md("Profil **portable** (sans `sttl`, `dttl`, `ct_state_ttl`) : modèles principaux uniquement, recherche allégée."),
    code("""
t_bin_p = TE.train_task('binary', 'portable', reg, curves, train, test, skip_done=RESUME)
t_multi_p = TE.train_task('multiclass', 'portable', reg, curves, train, test, skip_done=RESUME)
"""),
    md("""
## 3. Sauvegarde des modèles déployés

Le meilleur pipeline **selon la CV** est copié dans `models/binaire/pipeline.joblib` et
`models/multiclasse/pipeline.joblib`, avec un `metadata.json` (features, classes, hyperparamètres, scores CV et test,
versions). Ce sont ces fichiers que chargent l'application web et `scripts/predict_pcap.py`.
"""),
    code("""
deployed = TE.deploy(reg)
display(pd.DataFrame({t: {k: m[k] for k in ('model_name', 'cv_score', 'deployed')} for t, m in deployed.items()}).T)
"""),
    md("## 4. Tableau comparatif — tâche binaire (profil complet)"),
    code("""
def table(task, profile='full'):
    t = pd.DataFrame(reg['metrics'][profile][task]).drop(columns=['per_class'], errors='ignore')
    return t.set_index('model')
bt = table('binary')
cols = ['family', 'cv_common_mean', 'cv_mean', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'mcc', 'false_alarm_rate', 'n_configs', 'fit_time_s']
display(bt[cols].sort_values('f1', ascending=False).round(4))
print('Modèle déployé (meilleure moyenne sur les plis communs) :', reg['best']['full']['binary'])
print("cv_common_mean = moyenne sur 5 plis communs à tous les modèles (critère de sélection) ; cv_mean = score de la recherche d'hyperparamètres")
"""),
    code("""
s = bt.drop(index=[i for i in bt.index if 'Zero-R' in i]).sort_values('f1')
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(s)); h = 0.2
for k, (m, lab) in enumerate([('f1', 'F1'), ('recall', 'Rappel'), ('precision', 'Précision'), ('roc_auc', 'ROC-AUC')]):
    ax.barh(y + (k - 1.5) * h, s[m], height=h, label=lab, color=V.SERIES[k])
ax.set_yticks(y, s.index); ax.set_xlim(max(0, s.f1.min() - 0.1), 1.0); ax.legend(loc='lower right')
ax.set_title('Comparaison des modèles — détection binaire (test)'); V.save(fig, '03_model_comparison_binary'); plt.show()
"""),
    md("## 5. Courbes ROC et Précision-Rappel"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
order = bt.sort_values('roc_auc', ascending=False)
for i, (name, r) in enumerate(order.iterrows()):
    c = curves.get(f"binary|full|{r['key']}", {})
    if 'roc' not in c: continue
    col = V.SERIES[i % 8] if i < 8 else '#b0b0b0'
    axes[0].plot(c['roc']['fpr'], c['roc']['tpr'], color=col, lw=1.6, label=f"{name} ({r.roc_auc:.4f})")
    axes[1].plot(c['pr']['recall'], c['pr']['precision'], color=col, lw=1.6, label=f"{name} ({r.pr_auc:.4f})")
axes[0].plot([0, 1], [0, 1], ':', color='#898781'); axes[0].set_xlabel('Taux de faux positifs'); axes[0].set_ylabel('Rappel')
axes[0].set_title('ROC (AUC)'); axes[0].legend(fontsize=7, loc='lower right')
axes[1].set_xlabel('Rappel'); axes[1].set_ylabel('Précision'); axes[1].set_title('Précision-Rappel (AP)'); axes[1].legend(fontsize=7, loc='lower left')
axes[0].set_xlim(0, 0.3); axes[0].set_ylim(0.7, 1.005); axes[1].set_ylim(0.7, 1.005)
plt.tight_layout(); V.save(fig, '03_roc_pr_curves'); plt.show()
print('Zoom sur la zone utile (FPR < 30 %) : toutes les courbes sont proches du coin supérieur gauche.')
"""),
    md("## 6. Grille des matrices de confusion (binaire)"),
    code("""
keys = bt.sort_values('f1', ascending=False)
n = len(keys); ncol = 5; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3 * nrow))
for ax, (name, r) in zip(axes.ravel(), keys.iterrows()):
    cm = np.array(curves[f"binary|full|{r['key']}"]['confusion'])
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt='d', cmap=V.SEQ_CMAP, vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=['N', 'A'], yticklabels=['N', 'A'], annot_kws={'size': 8})
    ax.set_title(f"{name}\\nF1={r.f1:.4f}", fontsize=8)
for ax in axes.ravel()[n:]: ax.axis('off')
plt.tight_layout(); V.save(fig, '03_confusion_grid_binary'); plt.show()
"""),
    md("""
## 7. Analyse par famille d'algorithmes (résultats des recherches d'hyperparamètres)

Les tables complètes de la CV sont dans `results/tables/cv_binary_full_<modèle>.csv`.
"""),
    code("""
def cv(key, task='binary'):
    return pd.read_csv(C.METRICS_DIR / f'cv_{task}_full_{key}.csv')
k = cv('knn'); dt = cv('decision_tree'); nb = cv('bernoulli_nb'); svm = cv('svm')
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for (p, w), g in k.groupby(['param_clf__p', 'param_clf__weights']):
    axes[0].plot(g.param_clf__n_neighbors, g.mean_test_f1, marker='o', label=f"p={p}, {w}")
axes[0].set_xlabel('K'); axes[0].set_title('K-NN : F1 CV selon K'); axes[0].legend(fontsize=7)
for crit, g in dt.groupby('param_clf__criterion'):
    gg = g.groupby(g.param_clf__max_depth.fillna(99))['mean_test_f1'].max()
    axes[1].plot(gg.index.astype(int), gg.values, marker='o', label=crit)
axes[1].set_xlabel('max_depth (99 = illimitée)'); axes[1].set_title('Arbre : Gini vs entropie'); axes[1].legend()
for b, g in nb.groupby('param_clf__binarize'):
    axes[2].semilogx(g.param_clf__alpha, g.mean_test_f1, marker='o', label=f'binarize={b}')
axes[2].set_xlabel('α (lissage de Laplace)'); axes[2].set_title('Bernoulli NB'); axes[2].legend()
sv = svm.groupby('param_clf__kernel')['mean_test_f1'].max()
axes[3].bar(sv.index, sv.values, color=V.SERIES[:len(sv)]); axes[3].set_ylim(sv.min() - 0.02, 1)
axes[3].set_title('SVM : meilleur F1 CV par noyau')
plt.tight_layout(); V.save(fig, '03_hyperparameter_studies'); plt.show()
for key in ['knn', 'decision_tree', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'svm', 'xgboost']:
    print(f"{key:<16}", reg['models']['full']['binary'][key]['params'])
"""),
    md("## 8. Tâche multi-classe (10 catégories)"),
    code("""
mt = table('multiclass')
display(mt[['family', 'cv_common_mean', 'cv_mean', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'roc_auc_ovr', 'mcc']]
        .sort_values('f1_macro', ascending=False).round(4))
best_m = reg['best']['full']['multiclass']
cm = np.array(curves[f'multiclass|full|{best_m}']['confusion'])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt='d', cmap=V.SEQ_CMAP, ax=axes[0], cbar=False,
            xticklabels=reg['class_names'], yticklabels=reg['class_names'], annot_kws={'size': 7})
axes[0].set_xlabel('Prédit'); axes[0].set_ylabel('Réel'); axes[0].set_title(f'Matrice de confusion — {best_m}')
pc = pd.DataFrame({r['model']: {c: v['f1-score'] for c, v in r['per_class'].items()} for r in reg['metrics']['full']['multiclass'] if r.get('per_class')}).T
sns.heatmap(pc.loc[pc.mean(1).sort_values(ascending=False).index], annot=True, fmt='.2f', cmap=V.SEQ_CMAP, vmin=0, vmax=1, ax=axes[1], cbar=False, annot_kws={'size': 7})
axes[1].set_title('F1 par catégorie'); plt.tight_layout(); V.save(fig, '03_multiclass_results'); plt.show()
"""),
    md("""
**Analyse des erreurs :** *Normal*, *Generic* et *Reconnaissance* sont bien reconnues ; **DoS, Exploits, Analysis
et Backdoor** sont confondues entre elles. Ce n'est pas un défaut des modèles : ces catégories partagent des
vecteurs de features **identiques** (cf. notebook 01 : 8 053 vecteurs en conflit d'étiquette). La thèse
de N. Moustafa et la littérature font le même constat. Le F1-macro est donc borné bien en dessous du F1 binaire.

## 9. Profil portable (sans TTL)
"""),
    code("""
pt = pd.concat({'binaire': table('binary', 'portable')[['cv_mean', 'f1', 'recall', 'roc_auc']],
                'multi-classe': table('multiclass', 'portable')[['cv_mean', 'f1_macro', 'accuracy']]}, axis=1)
display(pt.round(4))
"""),
    md("## 10. Choix du seuil de décision"),
    code("""
bk = reg['best']['full']['binary']
thr = pd.DataFrame(curves[f'binary|full|{bk}']['thresholds'])
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (c, l) in enumerate([('precision', 'Précision'), ('recall', 'Rappel'), ('f1', 'F1'), ('false_alarm_rate', 'Fausses alertes')]):
    ax.plot(thr.threshold, thr[c], marker='o', ms=3, label=l, color=V.SERIES[i])
ax.set_xlabel('Seuil sur P(attaque)'); ax.legend(); ax.set_title(f'Arbitrage rappel / fausses alertes — {bk}')
V.save(fig, '03_threshold_analysis'); plt.show()
display(thr.round(4).iloc[::3])
"""),
    md("""
## 11. Comparaison statistique des modèles

Une différence de 0,001 de F1 peut être due au hasard du découpage. On teste :
- **Friedman** (H₀ : tous les modèles ont le même rang moyen sur les 5 plis communs) puis distance critique de **Nemenyi** ;
- **Wilcoxon signé** meilleur vs chaque autre modèle, correction de **Holm** ;
- **McNemar** entre les deux meilleurs sur le test (H₀ : mêmes taux d'erreur sur les flux où ils divergent).
"""),
    code("""
from src import run_analysis as RA
stats = RA.model_statistics(test)          # Friedman + Nemenyi, Wilcoxon-Holm, McNemar
for task, st in stats.items():
    fr, mc = st['friedman'], st['mcnemar']
    print(f"[{task}] Friedman χ²={fr['chi2']:.2f}, p={fr['p_value']:.2e}, CD Nemenyi={fr['critical_difference']:.2f}")
    print(f"        McNemar {mc['model_a']} vs {mc['model_b']} : b={mc['b']}, c={mc['c']}, p={mc['p_value']:.3g}")
    display(pd.Series(fr['mean_ranks'], name='rang moyen').to_frame().T.round(2))
"""),
    md("## 12. Courbes d'apprentissage"),
    code("""
lc = RA.learning_curves(train)            # mis en cache dans reports/metrics/learning_curves.json
fig, ax = plt.subplots(figsize=(8, 4))
for i, (k, d) in enumerate(lc.items()):
    ax.plot(d['sizes'], d['val_mean'], marker='o', color=V.SERIES[i], label=f'{k} — validation')
    ax.plot(d['sizes'], d['train_mean'], ls=':', color=V.SERIES[i], label=f'{k} — entraînement')
ax.set_xlabel("Taille du train"); ax.set_ylabel('F1'); ax.legend(fontsize=8); ax.set_title("Courbes d'apprentissage")
V.save(fig, '03_learning_curves'); plt.show()
"""),
    md("""
## 13. Conclusion

- Tous les modèles battent largement **Zero-R** (F1 = 0 : il ne détecte aucune attaque).
- Les **ensembles d'arbres** (XGBoost, Random Forest, Extra Trees, HistGB, vote, stacking) dominent ; K-NN et
  l'arbre CART sont très proches. Les tests de Friedman/Nemenyi/McNemar indiquent quels écarts sont réels.
- Naïve Bayes est nettement en retrait : l'hypothèse d'indépendance conditionnelle est violée par des
  features très corrélées (cf. Spearman).
- Le **profil portable** perd peu en binaire : l'information n'est pas uniquement dans le TTL.
- Modèles déployés : `models/binaire/` et `models/multiclasse/` (meilleurs en CV), seuil réglable dans l'application.
"""),
]

# =============================================================================
# 04 — Non supervisé
# =============================================================================
nb04 = [
    md("""
# 04 — Apprentissage non supervisé : clustering et détection d'anomalies

**Rôle :** découvrir des groupes de flux **sans utiliser les étiquettes** (parcours B du cours), puis vérifier
a posteriori si ces groupes correspondent aux catégories d'attaque ; et détecter des intrusions comme des
**anomalies** par rapport au trafic normal (utile pour des attaques inconnues).

Algorithmes : **K-Means** (coude, k-means++), **K-Medoids PAM** et **CLARA** (implémentés), **AGNES**
(Ward, complète, moyenne, simple), **DIANA** (implémenté), **DBSCAN**, **GMM** ; détecteurs Isolation Forest,
LOF, distance au centroïde, log-vraisemblance GMM. Code : `src/unsupervised/clustering.py`, `src/run_unsupervised.py`.
"""),
    code(SETUP),
    code("""
RUN = False
from src import run_unsupervised as RU
if RUN or not (C.METRICS_DIR / 'unsupervised.json').exists():
    RU.main()
u = json.loads((C.METRICS_DIR / 'unsupervised.json').read_text(encoding='utf-8'))
print(f"Espace : 10 composantes principales ({sum(u['pca_explained']):.1%} de variance) ; k coude = {u['k_elbow']}, k silhouette = {u['k_silhouette']}")
"""),
    md("""
## 1. K-Means : choix de k
- **Méthode du coude** : l'inertie intra-classe (WCSS) diminue toujours avec k ; on cherche le point où le gain
  marginal s'effondre (détecté automatiquement par la distance maximale à la corde).
- **Silhouette** s(i) = (b − a)/max(a, b) ∈ [−1, 1] et **Davies-Bouldin** (↓) complètent le diagnostic.
- **k-means++** (initialisation dispersée) vs aléatoire.
"""),
    code("""
km = pd.DataFrame(u['kmeans_scan'])
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for i, (init, g) in enumerate(km.groupby('init')):
    axes[0].plot(g.k, g.inertia_wcss, marker='o', label=init, color=V.SERIES[i])
axes[0].axvline(u['k_elbow'], ls=':', color='#eda100'); axes[0].set_xlabel('k'); axes[0].set_ylabel('WCSS'); axes[0].legend(); axes[0].set_title('Méthode du coude')
g = km[km.init == 'k-means++']
axes[1].plot(g.k, g.silhouette, marker='o', label='Silhouette ↑', color=V.SERIES[0]); axes[1].plot(g.k, g.davies_bouldin, marker='s', label='Davies-Bouldin ↓', color=V.SERIES[1])
axes[1].set_xlabel('k'); axes[1].legend(); axes[1].set_title('Indices internes')
plt.tight_layout(); V.save(fig, '04_kmeans_elbow_silhouette'); plt.show()
"""),
    md("## 2. Comparaison de tous les algorithmes\nIndices **internes** (silhouette, DB, Calinski-Harabasz) et **externes** vs les 10 catégories réelles (Rand Index, ARI, NMI, pureté)."),
    code("""
cmp = pd.DataFrame(u['comparison'])
display(cmp[['algorithm', 'n_clusters', 'noise_pct', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'rand_index_cat', 'adjusted_rand_cat', 'nmi_cat', 'purity_cat', 'purity_binary']].round(3))
"""),
    md("## 3. Projection ACP et correspondance clusters / catégories"),
    code("""
P = u['pca_points']; pts = pd.DataFrame(P)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for k, g in pts.groupby('cluster'):
    axes[0].scatter(g.x, g.y, s=6, alpha=0.6, color=V.SERIES[k % 8], label=f'C{k}')
axes[0].legend(markerscale=3, fontsize=8); axes[0].set_title('Clusters K-Means (ACP)')
for lab, g in pts.groupby('label'):
    axes[1].scatter(g.x, g.y, s=6, alpha=0.5, color=V.NORMAL if lab == 0 else V.ATTACK, label='Normal' if lab == 0 else 'Attaque')
axes[1].legend(markerscale=3); axes[1].set_title('Classes réelles (même projection)')
ct = pd.DataFrame(u['cluster_vs_category']['values'], index=u['cluster_vs_category']['index'], columns=u['cluster_vs_category']['columns'])
sns.heatmap(ct.div(ct.sum(1), axis=0), annot=ct, fmt='d', cmap=V.SEQ_CMAP, ax=axes[2], cbar=False, annot_kws={'size': 7})
axes[2].set_title('Clusters × catégories'); axes[2].set_ylabel('cluster')
plt.tight_layout(); V.save(fig, '04_pca_clusters'); plt.show()
display(pd.DataFrame(u['profiles']).round(3))
"""),
    md("## 4. Clustering hiérarchique : AGNES (ascendant) et DIANA (descendant)\nLa **corrélation cophénétique** mesure la fidélité du dendrogramme aux distances d'origine."),
    code("""
import joblib
from scipy.cluster.hierarchy import dendrogram
lk = joblib.load(C.ALL_MODELS_DIR / 'linkages.joblib')
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, m in zip(axes, ['ward', 'complete', 'average', 'single']):
    dendrogram(lk[m], truncate_mode='lastp', p=20, ax=ax, color_threshold=None, above_threshold_color=V.SERIES[0], no_labels=True)
    ax.set_title(f'AGNES — liaison {m}')
plt.tight_layout(); V.save(fig, '04_dendrograms'); plt.show()
display(pd.DataFrame(u['agnes']).round(3))
print('DIANA — scissions :', u['diana_history'])
"""),
    md("""
**Lecture :** la liaison *simple* produit l'effet de chaîne (un énorme cluster + singletons) ; *Ward* donne des
groupes équilibrés. DIANA, descendant, isole d'abord le groupe le plus éloigné (les flux d'attaque atypiques).

## 5. DBSCAN et GMM
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(u['k_distance'], color=V.SERIES[0]); axes[0].set_title('Graphe des 10-distances (choix de ε)'); axes[0].set_xlabel('points triés')
gm = pd.DataFrame(u['gmm_scan'])
axes[1].plot(gm.k, gm.bic, marker='o', label='BIC'); axes[1].plot(gm.k, gm.aic, marker='s', label='AIC'); axes[1].legend(); axes[1].set_title('GMM : critères d\\'information'); axes[1].set_xlabel('composantes')
plt.tight_layout(); V.save(fig, '04_dbscan_gmm'); plt.show()
display(pd.DataFrame(u['dbscan_scan']).round(3))
"""),
    md("## 6. Robustesse : K-Medoids (PAM) vs K-Means"),
    code("""
print(u['robustness'])
print('Un medoïde est un point réel du jeu : les valeurs aberrantes injectées le déplacent moins qu\\'une moyenne.')
"""),
    md("## 7. Détection d'intrusions par anomalies"),
    code("""
an = pd.DataFrame(u['anomaly'])
display(an.round(4))
fig, ax = plt.subplots(figsize=(8, 3))
s = an.sort_values('roc_auc'); ax.barh(s.detector + ' — ' + s['mode'].str.split().str[0], s.roc_auc, color=V.SERIES[2])
ax.set_xlim(0.4, 1); ax.set_xlabel('ROC-AUC (test)'); ax.set_title('Détecteurs d\\'anomalies'); V.save(fig, '04_anomaly_detection'); plt.show()
"""),
    md("""
## 8. Conclusion
- Les clusters retrouvent la **séparation normal / attaque** (pureté binaire élevée) mais pas finement les
  10 catégories (ARI modéré) : sans étiquettes, DoS/Exploits/Analysis/Backdoor restent indiscernables.
- Le meilleur compromis interne/externe est obtenu par K-Means / Ward ; DBSCAN isole du bruit utile (flux rares).
- Les détecteurs appris sur le seul trafic normal atteignent une ROC-AUC élevée **sans jamais voir d'attaque** :
  complément intéressant d'un classifieur supervisé pour les attaques inédites.
"""),
]

# =============================================================================
# 05 — Règles d'association
# =============================================================================
nb05 = [
    md("""
# 05 — Règles d'association (Apriori)

**Rôle :** extraire des règles lisibles « *si* conditions *alors* classe » (chapitre 5 du cours) qui décrivent
des signatures d'intrusion. Implémentation d'Apriori : `src/unsupervised/association.py`.

1. **Discrétisation** : chaque variable numérique est découpée en terciles (bas / moyen / élevé) ; les
   catégorielles gardent leurs 6 modalités principales.
2. **Itemsets fréquents** : niveau par niveau ; un candidat de taille k n'est compté que si tous ses
   sous-ensembles de taille k−1 sont fréquents (**anti-monotonie** du support).
3. **Règles** X ⇒ Y avec support, confiance, lift, levier et conviction.
"""),
    code(SETUP),
    code("""
from src.unsupervised import association as AR
train = pd.read_csv(C.TRAIN_FILE, low_memory=False)
tr = AR.discretize(train.sample(6000, random_state=1))
print('Exemple de transaction :', sorted(tr[0]))
import time; t = time.time()
freq = AR.apriori(tr, min_support=0.05, max_len=4)
print(f"{len(freq)} itemsets fréquents (support ≥ 5 %) en {time.time() - t:.1f}s")
print(pd.Series({k: sum(1 for s in freq if len(s) == k) for k in range(1, 5)}, name="nombre d'itemsets par taille"))
"""),
    code("""
rules = AR.association_rules(freq, min_confidence=0.9, consequent_prefix='classe=')
print(len(rules), 'règles vers la classe (confiance ≥ 90 %)')
display(rules.head(15).round(4))
"""),
    md("## Règles vers les catégories d'attaque (calculées sur 12 000 attaques par `src.run_unsupervised`)"),
    code("""
rc = pd.read_csv(C.METRICS_DIR / 'association_rules_category.csv')
display(rc.head(15).round(4))
fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(rules.support, rules.confidence, c=rules.lift, cmap=V.SEQ_CMAP, s=25, edgecolor='#52514e', linewidth=0.3)
plt.colorbar(sc, label='lift'); ax.set_xlabel('support'); ax.set_ylabel('confiance'); ax.set_title('Règles ⇒ classe')
V.save(fig, '05_association_rules'); plt.show()
"""),
    md("""
**Lecture :** les règles de plus fort lift confirment l'EDA : `sttl=élevé ∧ dttl=bas ⇒ Attaque`,
`service=dns ∧ dbytes=bas ∧ ct_srv_src=élevé ⇒ Generic` (rafales DNS unidirectionnelles), `state=fin ∧ sttl=bas ⇒ Normal`.
Contrairement à un arbre, chaque règle est évaluée indépendamment : elles servent d'explications
globales, pas de classifieur complet.
"""),
]

# =============================================================================
# 06 — Explicabilité
# =============================================================================
nb06 = [
    md("""
# 06 — Importance des variables et explicabilité

**Rôle :** comprendre *pourquoi* les modèles décident, avec cinq méthodes (`src/run_analysis.py`) :

| Méthode | Principe | Limite |
|---|---|---|
| Impureté (Gini) — RF | baisse moyenne d'impureté apportée par la variable | biaisée vers les variables à nombreuses valeurs |
| Gain — XGBoost | amélioration moyenne de la perte aux splits | propre au modèle |
| **SHAP** (TreeSHAP) | valeur de Shapley : contribution exacte de chaque variable à chaque prédiction | coûteux hors arbres |
| Permutation | baisse de F1 sur le test quand on mélange la variable | sous-estime les variables corrélées |
| Information mutuelle | dépendance statistique variable ↔ cible, sans modèle | univariée |

Les valeurs SHAP sont calculées nativement par XGBoost (`pred_contribs=True`) : pas de dépendance supplémentaire.
"""),
    code(SETUP),
    code("""
RUN = False
from src import run_analysis as RA
if RUN or not (C.METRICS_DIR / 'analysis.json').exists():
    RA.main()
imp = pd.read_csv(C.METRICS_DIR / 'feature_importance.csv', index_col=0)
display(imp[['mean_rank', 'shap_xgboost', 'permutation_best', 'impurity_rf', 'gain_xgboost', 'mutual_info']].head(20).round(4))
"""),
    code("""
methods = [('shap_xgboost', 'SHAP |φ| moyen'), ('permutation_best', 'Permutation (Δ F1)'), ('impurity_rf', 'Gini (RF)'),
           ('gain_xgboost', 'Gain (XGBoost)'), ('mutual_info', 'Information mutuelle')]
fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
for ax, (m, l) in zip(axes, methods):
    s = imp[m].sort_values(ascending=False).head(15)
    V.barh(ax, s.index, s.values, fmt='{:.3f}'); ax.set_title(l)
plt.tight_layout(); V.save(fig, '06_feature_importance_methods'); plt.show()
agree = pd.read_csv(C.METRICS_DIR / 'feature_importance_rank_agreement.csv', index_col=0)
display(agree.round(2))
"""),
    md("## Résumé SHAP (style *beeswarm*) et explication locale"),
    code("""
import joblib
from src.evaluation import explain as EX
reg = json.loads((C.REGISTRY_FILE).read_text(encoding='utf-8'))
xgb = joblib.load(C.ALL_MODELS_DIR / reg['models']['full']['binary']['xgboost']['file'])
test = pd.read_csv(C.TEST_FILE, low_memory=False)
sub = test.sample(1500, random_state=3)
from src.preprocessing import feature_frame
vals, names, base = EX.shap_values(xgb, sub)
Z = xgb.named_steps['prep'].transform(feature_frame(sub))
order = np.argsort(np.abs(vals).mean(0))[::-1][:15]
fig, ax = plt.subplots(figsize=(9, 6))
for r, j in enumerate(order):
    z = Z[:, j]; zc = (z - z.min()) / (np.ptp(z) + 1e-9)
    ax.scatter(vals[:, j], np.full(len(z), r) + np.random.default_rng(r).normal(0, 0.12, len(z)), c=zc, cmap='coolwarm', s=5, alpha=0.7)
ax.set_yticks(range(len(order)), [names[j] for j in order]); ax.invert_yaxis(); ax.axvline(0, color='#898781', lw=0.8)
ax.set_xlabel('valeur SHAP (log-odds) — couleur : valeur de la variable (bleu bas, rouge haut)'); ax.set_title('Résumé SHAP — XGBoost binaire')
V.save(fig, '06_shap_summary'); plt.show()
"""),
    code("""
row = test[test.attack_cat == 'Exploits'].sample(1, random_state=0)
ex = EX.explain_instance(xgb, row, top=10)
c = pd.DataFrame(ex['contributions'])
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(c.feature + ' = ' + c.value.astype(str).str[:8], c.contribution, color=[V.ATTACK if v > 0 else V.NORMAL for v in c.contribution])
ax.invert_yaxis(); ax.axvline(0, color='#52514e', lw=0.8); ax.set_xlabel('contribution SHAP (log-odds)')
ax.set_title(f"Explication locale d'un flux Exploits — P(attaque) = {ex['probability']:.3f}")
V.save(fig, '06_shap_local'); plt.show()
print(EX.text_explanation(int(ex['probability'] > 0.5), ex['probability'], 'Exploits', ex['contributions']))
"""),
    md("## SHAP par catégorie et ablation par groupe de features (Tables 3.11 à 3.14 de la thèse)"),
    code("""
sh = pd.read_csv(C.METRICS_DIR / 'shap_multiclass_by_category.csv', index_col=0)
top = sh.sum(1).sort_values(ascending=False).index[:15]
fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
sns.heatmap(sh.loc[top], cmap=V.SEQ_CMAP, ax=axes[0], annot=True, fmt='.2f', annot_kws={'size': 7}, cbar=False)
axes[0].set_title('|SHAP| moyen par catégorie (XGBoost multi-classe)')
ab = pd.read_csv(C.METRICS_DIR / 'ablation_feature_groups.csv')
y = np.arange(len(ab))
axes[1].barh(y - 0.2, ab.delta_binary, 0.4, label='Δ F1 binaire', color=V.SERIES[0]); axes[1].barh(y + 0.2, ab.delta_multiclass, 0.4, label='Δ F1-macro', color=V.SERIES[1])
axes[1].set_yticks(y, ab.removed_group); axes[1].invert_yaxis(); axes[1].axvline(0, color='#52514e', lw=0.8); axes[1].legend(); axes[1].set_title('Ablation : retrait d\\'un groupe')
plt.tight_layout(); V.save(fig, '06_shap_category_ablation'); plt.show()
display(ab.round(4))
"""),
    md("""
**Conclusions :**
- Consensus des cinq méthodes : `sttl` / `ct_state_ttl`, la taille moyenne des paquets (`smeansz`, `dmeansz`),
  les volumes (`sbytes`, `dbytes`), les compteurs de connexion `ct_*` et `service`.
- SHAP par catégorie : *Generic* s'appuie sur `ct_srv_src`/`service`, *Reconnaissance* sur l'état et la durée, etc.
- L'ablation montre la **redondance** : retirer un groupe coûte peu car d'autres variables portent la même
  information — sauf pour la tâche multi-classe, plus exigeante.
"""),
]

# =============================================================================
# 07 — Extracteur PCAP
# =============================================================================
nb07 = [
    md("""
# 07 — Génération des tuples UNSW-NB15 depuis un fichier PCAP

**Problème :** un modèle entraîné sur les 49 colonnes du CSV UNSW-NB15 est inutilisable sur une vraie capture
si l'on ne sait pas **calculer ces colonnes** à partir des paquets. La thèse de N. Moustafa (§3.5, Figure 3.4)
décrit la chaîne d'origine : Argus (flux) + Bro-IDS (journaux CONN/HTTP/FTP) + 12 features additionnelles
(Algorithmes 3.2 à 3.9), le tout appareillé sur le 5-tuple.

**Notre solution :** une réimplémentation en Python pur (`src/pcap/`) :

| Module | Rôle (équivalent thèse) |
|---|---|
| `reader.py` | lecture pcap / pcapng, décodage Ethernet/VLAN/SLL → IPv4/IPv6 → TCP/UDP/ICMP/ARP |
| `flows.py` | flux bidirectionnels **Argus** : état, octets, TTL, pertes, débits, gigue, RTT ; features additionnelles |
| `app_layer.py` | **Bro-IDS** : détection de service par signature, HTTP (trans_depth, res_bdy_len), FTP |
| `generator.py` | captures de démonstration avec vérité terrain par flux |
"""),
    code(SETUP),
    md("""
## 1. Validation des formules sur le CSV officiel

Avant d'écrire l'extracteur, nous avons **retrouvé empiriquement** les formules d'Argus en les confrontant
aux valeurs publiées (200 000 flux du fichier 1). Exemple : `sload` n'est pas `sbytes·8/dur` mais
`(sbytes − sbytes/spkts)·8/dur` — Argus ignore un paquet dans le calcul du débit.
"""),
    code("""
from src.run_analysis import extractor_fidelity
fid = extractor_fidelity()
"""),
    code("""
from src.preprocessing import read_raw_file, clean_dataframe
d = clean_dataframe(read_raw_file(C.RAW_FILES[0], nrows=200000))
d = d[(d.dur > 0) & (d.spkts > 1)]
naive = (np.abs(d.sbytes * 8 / d.dur - d.sload) <= 0.01 * d.sload).mean()
argus = (np.abs((d.sbytes - d.sbytes / d.spkts) * 8 / d.dur - d.sload) <= 0.01 * d.sload).mean()
print(f"sload = sbytes·8/dur : {naive:.1%} de concordance   |   (sbytes − sbytes/spkts)·8/dur : {argus:.1%}")
"""),
    md("""
**Limite assumée :** les compteurs `ct_*` (fenêtres de 100 connexions) ne se reproduisent qu'approximativement :
le CSV publié ne contient qu'un sous-ensemble des enregistrements Argus et leur ordre exact n'est pas documenté.
Nous implémentons la lecture « blocs de 100 enregistrements triés par `ltime` » de l'Algorithme 3.7
(la plus proche des valeurs publiées) et une variante en fenêtre glissante.

## 2. Captures de démonstration

Faute de pouvoir versionner les 100 Go de PCAP originaux, `generator.py` synthétise du trafic reproduisant la
topologie du banc de test (clients 59.166.0.0/24 en TTL 31, serveurs 149.171.126.0/24, attaquants
175.45.176.0/24 en TTL 254/62) : HTTP, DNS, FTP (+ ftp-data), SSH, SMTP et les 9 familles d'attaques, avec
une **vérité terrain par flux**. Une seconde capture utilise des TTL d'hôtes modernes (64/128).
"""),
    code("""
from src.pcap.generator import generate_demo_pcap, match_ground_truth
from src.pcap import pcap_to_flows
for name, kw in [('demo_unsw_testbed', {}), ('demo_modern_ttl', dict(seed=7, testbed_ttl=False, n_normal=120, n_attack_each=8))]:
    p = C.SAMPLE_PCAP_DIR / f'{name}.pcap'
    if not p.exists():
        generate_demo_pcap(p, **kw)
    print(name, f"{p.stat().st_size / 1e6:.2f} Mo")
import time; t = time.time()
flows = pcap_to_flows(C.SAMPLE_PCAP_DIR / 'demo_unsw_testbed.pcap')
print(f"{flows.attrs['n_packets']:,} paquets → {len(flows)} tuples de {flows.shape[1]} features en {time.time() - t:.2f}s")
display(flows.head(8))
"""),
    code("""
lab = pd.read_csv(C.SAMPLE_PCAP_DIR / 'demo_unsw_testbed_labels.csv')
m = match_ground_truth(flows, lab)
display(m.groupby('true_attack_cat')[['proto', 'state', 'service']].agg(lambda s: s.value_counts().index[0]))
display(m.groupby('true_attack_cat')[['dur', 'sbytes', 'dbytes', 'spkts', 'sttl', 'dttl', 'ct_state_ttl', 'smeansz', 'ct_srv_src']].median().round(4))
"""),
    md("""
## 3. Test aller-retour sur du trafic réel : rejeu d'enregistrements UNSW-NB15

Les scénarios ci-dessus sont écrits à la main : leurs « attaques » ne reproduisent pas le comportement
des attaques générées par IXIA dans le banc de test. Pour valider la chaîne sur un trafic **réel**, on rejoue
une tranche contiguë de 2 500 enregistrements du CSV officiel (fichier 4, toutes les familles d'attaques) :
pour chaque enregistrement, `src/pcap/replay.py` reconstruit les paquets qui le produisent (mêmes IP,
ports, TTL, nombres de paquets et d'octets, durée, délais de poignée de main, état, signature applicative).
Le PCAP est tronqué à 160 octets par trame (comme `tcpdump -s 160`) : l'extracteur lit les longueurs dans
les en-têtes IP, ce qui le rend aussi robuste aux vraies captures tronquées.

**Question 1 — fidélité :** les features ré-extraites sont-elles égales aux valeurs d'origine ?
"""),
    code("""
from src.pcap.replay import build_replay_demos, roundtrip_fidelity
if not (C.SAMPLE_PCAP_DIR / 'demo_unsw_replay.pcap').exists():
    build_replay_demos()
rec = pd.read_csv(C.SAMPLE_PCAP_DIR / 'demo_unsw_replay_records.csv', low_memory=False)
fl_r = pcap_to_flows(C.SAMPLE_PCAP_DIR / 'demo_unsw_replay.pcap')
print(f"{len(rec)} enregistrements rejoués → {fl_r.attrs['n_packets']:,} paquets → {len(fl_r)} flux ré-extraits")
fid_rt = roundtrip_fidelity(rec, fl_r)
display(fid_rt)
fig, ax = plt.subplots(figsize=(8, 5))
s_ = fid_rt.sort_values('concordance_pct')
ax.barh(s_.feature, s_.concordance_pct, color=V.SERIES[2]); ax.set_xlim(80, 100.5)
ax.set_xlabel("% de flux dont la feature ré-extraite = valeur d'origine"); ax.set_title('Test aller-retour : enregistrement → PCAP → features')
V.save(fig, '07_replay_fidelity'); plt.show()
"""),
    md("""
**Lecture :** état, nombres de paquets, TTL, durée, profondeur HTTP, taille des corps de réponse, `ct_state_ttl`
sont reproduits à 99–100 %. Les écarts restants (≈ 5 %) viennent de flux TCP atypiques (retransmissions,
poignées de main incomplètes) que le rejeu simplifie. Les flux `unas` sont moins nombreux que les
enregistrements : Argus nomme « unas » tous les numéros de protocole IP non assignés ; le balayage de
protocoles du banc de test réutilise ces 110 numéros et les répétitions rapprochées fusionnent.

**Question 2 — détection de bout en bout, et artefact TTL.** On évalue les modèles déployés sur le rejeu,
puis sur le **même trafic** dont les TTL sont remplacés par 64 (hôtes modernes). On distingue les flux dont
l'enregistrement d'origine n'a **jamais été vu à l'entraînement**.
"""),
    code("""
from src import run_analysis as RA
e2e = pd.DataFrame(RA.pcap_end_to_end())
display(e2e[['description', 'profile', 'subset', 'n_flows', 'n_attacks', 'precision', 'recall', 'f1', 'false_alarm_rate', 'attack_type_accuracy']].round(3))
rp = e2e[e2e.pcap.str.startswith('demo_unsw_replay') & (e2e.subset == 'tous les flux')]
fig, ax = plt.subplots(figsize=(8, 3.5))
lab_ = rp.description.str.replace("Rejeu d'enregistrements réels UNSW-NB15 ", 'Rejeu ').str.replace('Même rejeu, ', 'Rejeu, ') + ' — ' + rp.profile.map({'full': 'complet', 'portable': 'portable'})
ax.barh(lab_, rp.recall, color=[V.SERIES[0] if p == 'full' else V.SERIES[2] for p in rp.profile])
ax.set_xlim(0, 1.05); ax.set_xlabel('Rappel (attaques détectées)'); ax.invert_yaxis(); ax.set_title('Artefact TTL : même trafic, TTL changé')
V.save(fig, '07_ttl_artifact'); plt.show()
"""),
    md("""
**Lecture :**
- Sur le trafic rejoué avec ses TTL d'origine, la chaîne complète détecte les attaques (rappel ≈ 0,9–1,0) et donne le
  **type d'attaque** exact pour la grande majorité d'entre elles, y compris sur les flux jamais vus.
- Avec des TTL de 64, le profil **complet s'effondre** (il a appris « TTL 254 ⇒ attaque », propriété du banc de test),
  alors que le profil **portable** garde exactement les mêmes performances : c'est le profil à utiliser sur un autre réseau.
- Les scénarios synthétiques, dont les attaques sont inventées, sont nettement moins bien reconnus : un modèle
  appris sur UNSW-NB15 reconnaît les attaques *du banc de test*, pas n'importe quel trafic anormal (d'où l'intérêt
  complémentaire des détecteurs d'anomalies du notebook 04).

## 4. Utilisation sur les PCAP originaux de UNSW-NB15
Les 100 Go de PCAP sont téléchargeables sur le site officiel. Pour un fichier :
```python
flows = pcap_to_flows('chemin/vers/1.pcap')     # 47 colonnes, ordre officiel
flows.to_csv('1_tuples.csv', index=False)
```
ou directement via la page **Analyse PCAP** de l'application (téléversement jusqu'à 200 Mo).
"""),
]

NOTEBOOKS = {
    "01_eda_and_cleaning.ipynb": nb01, "02_preprocessing_encoding.ipynb": nb02,
    "03_model_training_evaluation.ipynb": nb03, "04_clustering.ipynb": nb04,
    "05_regles_association.ipynb": nb05, "06_explicabilite.ipynb": nb06,
    "07_extraction_pcap.ipynb": nb07,
}

if __name__ == "__main__":
    # python notebooks/build_notebooks.py [03 07 ...]  -> ne régénère que ces notebooks
    import sys
    wanted = sys.argv[1:]
    for name, cells in NOTEBOOKS.items():
        if not wanted or any(name.startswith(w) for w in wanted):
            save(name, cells)
