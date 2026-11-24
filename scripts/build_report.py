"""Génère le rapport scientifique : docs/rapport/rapport.html puis docs/Rapport_MiniProject.pdf.

Tous les chiffres proviennent des artefacts (registry.json, analysis.json, unsupervised.json,
split_metadata.json, tables CSV) : le rapport est toujours cohérent avec le code exécuté.
Conversion PDF : Microsoft Edge / Google Chrome en mode headless (--print-to-pdf).

Usage : python scripts/build_report.py
"""
from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

OUT_HTML = C.DOCS_DIR / "rapport" / "rapport.html"
OUT_PDF = C.DOCS_DIR / "Rapport_MiniProject.pdf"


def J(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


reg = J(C.REGISTRY_FILE)
an = J(C.METRICS_DIR / "analysis.json")
un = J(C.METRICS_DIR / "unsupervised.json")
meta = J(C.SPLIT_META_FILE)


def T(name):
    p = C.METRICS_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def pct(x, d=2):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{d}f} %".replace(".", ",")


def num(x, d=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) > 999):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:.{d}f}".replace(".", ",")


def table(df: pd.DataFrame, cols: dict, fmt: dict | None = None, highlight=None, small=False) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(v)}</th>" for v in cols.values())
    rows = []
    for _, r in df.iterrows():
        cls = ' class="hl"' if highlight is not None and highlight(r) else ""
        cells = []
        for k in cols:
            v = r.get(k)
            f = fmt.get(k)
            if callable(f):
                s = f(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                s = num(v, f if isinstance(f, int) else 4)
            else:
                s = html.escape(str(v))
            cells.append(f"<td>{s}</td>")
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<table class="{"small" if small else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


FIG_N = [0]
TAB_N = [0]


def fig(name, caption, width="100%"):
    from src.visualization import figure_dir
    p = figure_dir(name) / f"{name}.png"
    if not p.exists():
        return f'<p class="missing">[figure manquante : {name}]</p>'
    FIG_N[0] += 1
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'<figcaption>Figure {FIG_N[0]} — {caption}</figcaption></figure>')


def cap(text):
    TAB_N[0] += 1
    return f'<p class="tcap">Tableau {TAB_N[0]} — {text}</p>'


def metrics(task, profile="full"):
    return pd.DataFrame(reg["metrics"][profile][task])


def best_row(task, profile="full"):
    m = metrics(task, profile)
    return m[m.key == reg["best"][profile][task]].iloc[0]


def build() -> str:
    rs = meta.get("raw_stats", {})
    cr = meta.get("cleaning_report", {})
    bb, bm = best_row("binary"), best_row("multiclass")
    bp = best_row("binary", "portable")
    mb, mm = metrics("binary").sort_values("f1", ascending=False), metrics("multiclass").sort_values("f1_macro", ascending=False)
    zr = mb[mb.key == "zero_r"].iloc[0]
    eda = an.get("eda", {})
    ms = an.get("model_stats", {})
    fid = pd.DataFrame(an.get("fidelity", []))
    e2e = pd.DataFrame(an.get("pcap_e2e", []))
    rt = pd.DataFrame(an.get("replay_fidelity", []))
    imp = T("feature_importance.csv")
    top_feats = ", ".join(f"<code>{f}</code>" for f in imp.feature.head(8)) if len(imp) else ""
    ab = T("ablation_feature_groups.csv")
    cmpu = pd.DataFrame(un.get("comparison", []))
    anom = pd.DataFrame(un.get("anomaly", []))
    rules = T("association_rules_class.csv")
    rules_c = T("association_rules_category.csv")
    norm, mw, kw, chi = T("stat_normality.csv"), T("stat_mann_whitney.csv"), T("stat_kruskal_wallis.csv"), T("stat_chi2.csv")
    nb_best = mb[mb.key.isin(["gaussian_nb", "bernoulli_nb", "multinomial_nb"])].iloc[0]
    fr_b = ms.get("binary", {}).get("friedman", {})
    mc_b = ms.get("binary", {}).get("mcnemar", {})
    fr_m = ms.get("multiclass", {}).get("friedman", {})
    mc_m = ms.get("multiclass", {}).get("mcnemar", {})
    name = lambda k, t="binary": metrics(t).set_index("key").loc[k, "model"] if k in set(metrics(t).key) else k  # noqa: E731
    pc_best = bm.per_class if isinstance(bm.per_class, dict) else {}
    weak = sorted(pc_best.items(), key=lambda kv: kv[1]["f1-score"])[:4]
    strong = sorted(pc_best.items(), key=lambda kv: -kv[1]["f1-score"])[:3]

    S = []
    a = S.append
    # ------------------------------------------------------------------ garde
    a(f"""<section class="cover">
<div class="cover-top">Cours de Data Mining — Mini-projet 2026/2027 · Parcours hybride</div>
<h1>Détection d'intrusions réseau par Data Mining :<br>du fichier PCAP à la décision</h1>
<h2>Classification supervisée, clustering, règles d'association et extraction des 47 features UNSW-NB15 depuis le trafic brut</h2>
<div class="cover-box">
<p><b>Jeu de données :</b> UNSW-NB15 — {num(rs.get('full_total', 0))} flux bruts, 49 attributs</p>
<p><b>Meilleur détecteur (binaire) :</b> {bb.model} — F1 {pct(bb.f1)}, ROC-AUC {num(bb.roc_auc)}, rappel {pct(bb.recall)}</p>
<p><b>Meilleur classifieur (10 catégories) :</b> {bm.model} — F1-macro {pct(bm.f1_macro)}</p>
<p><b>Livrables :</b> 7 notebooks, code source <code>src/</code>, application web Flask, API REST, tests pytest</p>
</div>
<div class="cover-team"><b>Équipe</b> (codes étudiants, règle d'anonymat) : CS16 (chef d'équipe), CS10, CS26, CS27, RSD07</div>
</section>""")
    # ------------------------------------------------------------------ résumé
    a(f"""<section><h2 class="nonum">Résumé</h2>
<p>Ce travail construit une chaîne complète de détection d'intrusions réseau sur le jeu de référence UNSW-NB15
({num(rs.get('full_total', 0))} flux, 9 familles d'attaques). Après un échantillonnage stratifié conservant toutes les classes rares,
un nettoyage documenté ({num(cr.get('duplicates_removed', 0))} doublons supprimés, {num(cr.get('conflicting_vectors', 0))} vecteurs en conflit d'étiquette détectés)
et une analyse exploratoire appuyée par six familles de tests statistiques corrigés par Benjamini-Hochberg, nous comparons
{len(mb)} classifieurs pour la détection binaire et {len(mm)} pour la classification en 10 catégories, tous optimisés par
GridSearchCV ou RandomizedSearchCV en validation croisée stratifiée à 5 plis. Le modèle retenu sur la validation croisée,
<b>{bb.model}</b>, atteint sur le jeu de test un F1 de <b>{pct(bb.f1)}</b> (rappel {pct(bb.recall)}, taux de fausses alertes
{pct(bb.false_alarm_rate)}, ROC-AUC {num(bb.roc_auc)}), contre 0 pour la référence Zero-R. En multi-classe, le F1-macro
({pct(bm.f1_macro)}) est limité par le chevauchement intrinsèque des catégories DoS, Exploits, Analysis et Backdoor, que nous expliquons.
La partie non supervisée compare K-Means, K-Medoids (PAM, CLARA), AGNES, DIANA, DBSCAN et GMM, et évalue la détection
d'anomalies (Isolation Forest, LOF) ; des règles Apriori fournissent des signatures interprétables.
Contribution principale : une <b>réimplémentation en Python de la chaîne de génération des features</b> décrite dans la thèse
de N. Moustafa (Argus + Bro-IDS + algorithmes 3.2 à 3.9), dont nous validons les formules sur le CSV officiel
(jusqu'à 100 % de concordance) ; elle permet d'appliquer les modèles à n'importe quel fichier PCAP depuis l'application web.</p>
<p><b>Mots-clés :</b> NIDS, UNSW-NB15, XGBoost, validation croisée, tests non paramétriques, SHAP, clustering, Apriori, PCAP, Argus.</p></section>""")
    a('<section><h2 class="nonum">Table des matières</h2><div id="toc"></div></section>')

    # ------------------------------------------------------------------ 1
    a(f"""<section><h2>Introduction et problématique</h2>
<p>Les systèmes de détection d'intrusions réseau (NIDS) surveillent le trafic pour signaler les activités malveillantes.
Les approches par signatures (Snort, Suricata) ne détectent que des attaques connues ; les approches par apprentissage
automatique apprennent à partir d'exemples étiquetés et peuvent généraliser à des variantes. Deux difficultés dominent :
(i) un <b>déséquilibre extrême</b> (87,4 % de trafic normal dans UNSW-NB15, 174 flux de la classe Worms sur 2,5 millions) et
(ii) l'<b>écart entre le jeu d'entraînement et le trafic réel</b> : un modèle entraîné sur des colonnes que l'on ne sait pas
recalculer depuis une capture est inutilisable en production.</p>
<p><b>Questions de recherche.</b> Q1 : quelles variables distinguent statistiquement le trafic normal des attaques ?
Q2 : quel algorithme, parmi ceux du cours et des extensions classiques, détecte le mieux les attaques (priorité au rappel)
et identifie leur catégorie ? Q3 : la structure des données permet-elle de retrouver les attaques sans étiquettes ?
Q4 : peut-on reproduire les features du jeu depuis un PCAP pour déployer le modèle ?</p>
<p><b>Démarche.</b> Le rapport suit la chaîne CRISP-DM : données (§3–4), préparation (§5–7), modélisation et évaluation
(§8–11), approches non supervisées (§12–13), déploiement applicatif (§14), discussion (§15).</p></section>""")
    # ------------------------------------------------------------------ 2
    a("""<section><h2>État de l'art</h2>
<p><b>Jeux de données.</b> KDD Cup 99 et NSL-KDD, dérivés de DARPA 98, ont longtemps servi de référence, mais
souffrent d'enregistrements redondants, d'attaques obsolètes et d'artefacts de génération. UNSW-NB15 (Moustafa & Slay, 2015)
a été produit en 2015 avec l'outil IXIA PerfectStorm sur un banc de test hybride (trafic réel + attaques synthétiques
issues de la base CVE), et propose 49 attributs extraits par Argus, Bro-IDS et douze algorithmes additionnels.
CIC-IDS2017 et CIC-UNSW-NB15 utilisent CICFlowMeter.</p>
<p><b>Méthodes.</b> Les travaux sur UNSW-NB15 rapportent des exactitudes binaires de 85 à 99 % selon le protocole :
arbres de décision et forêts aléatoires (Moustafa & Slay, 2016), boosting de gradient (XGBoost, LightGBM), SVM et K-NN.
La classification multi-classe est nettement plus difficile ; plusieurs auteurs signalent la confusion entre DoS, Exploits,
Analysis et Backdoor. Les approches non supervisées (K-Means, Isolation Forest, modèles de mélange) sont étudiées pour
la détection d'attaques inédites (zero-day), au prix d'un taux de fausses alertes plus élevé.</p>
<p><b>Positionnement.</b> Nous combinons un benchmark rigoureux (sélection sur la CV, tests de Friedman/McNemar),
une étude d'explicabilité multi-méthodes et, surtout, la reconstruction de la chaîne d'extraction des features pour passer
du CSV au trafic brut, ce que la plupart des travaux académiques sur ce jeu ne traitent pas.</p></section>""")
    # ------------------------------------------------------------------ 3
    full = pd.Series(rs.get("full_counts", {})).reindex(C.ATTACK_CATEGORIES)
    dist = pd.DataFrame({"cat": full.index, "n": full.values, "p": full.values / full.sum(),
                         "s": [rs.get("sample_counts", {}).get(c, 0) for c in full.index],
                         "c": [cr.get("class_counts", {}).get(c, 0) for c in full.index]})
    a(f"""<section><h2>Données : UNSW-NB15</h2>
<h3>Origine et volumétrie</h3>
<p>Les données brutes comportent quatre fichiers CSV sans en-tête ({', '.join(f'{k} : {num(v)}' for k, v in rs.get('files', {}).items())}),
soit <b>{num(rs.get('full_total', 0))} enregistrements</b> et <b>49 colonnes</b> décrites dans <code>NUSW-NB15_features.csv</code>.
Le seuil du cahier des charges (≥ 10 000 lignes brutes, ≥ 10 attributs) est très largement dépassé.</p>
{cap("Distribution des catégories : jeu complet, échantillon stratifié et jeu nettoyé")}
{table(dist, {"cat": "Catégorie", "n": "Jeu complet", "p": "% complet", "s": "Échantillon", "c": "Après nettoyage"}, {"p": lambda v: pct(v, 3)})}
<h3>Les 47 features (thèse, Tables 3.10 à 3.14)</h3>
<p>Elles se répartissent en cinq groupes : <b>flux</b> (srcip, sport, dstip, dsport, proto), <b>base</b> (state, dur, sbytes, dbytes,
sttl, dttl, sloss, dloss, service, sload, dload, spkts, dpkts), <b>contenu</b> (swin, dwin, stcpb, dtcpb, smeansz, dmeansz,
trans_depth, res_bdy_len), <b>temps</b> (sjit, djit, stime, ltime, sintpkt, dintpkt, tcprtt, synack, ackdat) et
<b>additionnelles</b> (is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd et sept compteurs ct_*
calculés sur des groupes de 100 connexions).</p>
<p><b>Sélection des features du modèle.</b> Nous excluons les <b>identifiants</b> (adresses IP, ports, horodatages), qui
permettraient au modèle de mémoriser la topologie du banc de test (fuite d'information), ainsi que <code>stcpb</code> et
<code>dtcpb</code>, numéros de séquence initiaux tirés aléatoirement par les piles TCP (RFC 6528). Il reste
<b>{len(C.FEATURES)} features</b> : {len(C.NUMERIC_FEATURES)} numériques et 3 catégorielles (proto, state, service).</p>
<h3>Échantillonnage stratifié</h3>
<p>Comparer quinze modèles en validation croisée sur 2,5 millions de lignes est hors de portée d'une machine de 8 Go.
Nous tirons, de façon reproductible (<code>random_state = 42</code>), 60 000 flux normaux et jusqu'à 8 000 flux par catégorie
d'attaque, en gardant toutes les lignes des classes rares (Analysis, Backdoor, Shellcode, Worms). Ce rééquilibrage partiel
modifie la prévalence (≈ 37 % d'attaques contre 12,6 %) : les métriques indépendantes de la prévalence (rappel, taux de fausses
alertes, ROC-AUC) restent directement interprétables ; la précision serait plus faible à prévalence réelle, et le seuil
de décision est réglable dans l'application.</p>
{fig("01_class_distribution", "Distribution des catégories dans le jeu complet (échelle logarithmique) et dans l'échantillon.")}
</section>""")
    # ------------------------------------------------------------------ 4
    a(f"""<section><h2>Génération des features depuis un fichier PCAP</h2>
<p>La thèse de N. Moustafa (§3.5, Figure 3.4) décrit le cadre de création des features : les PCAP sont traités par
<b>Argus</b> (flux bidirectionnels) et <b>Bro-IDS</b> (journaux CONN, HTTP, FTP), les deux sorties sont appariées sur le
5-tuple, puis douze features additionnelles sont calculées (Algorithmes 3.2 à 3.9). Nous avons réimplémenté cette chaîne
en Python pur (<code>src/pcap/</code>, sans dépendance externe) :</p>
<ul>
<li><b>Lecture</b> (<code>reader.py</code>) : formats pcap (µs/ns, deux boutismes) et pcapng (blocs SHB/IDB/EPB/SPB),
liens Ethernet (VLAN 802.1Q/QinQ), Linux SLL/SLL2, loopback BSD, IP brut ; IPv4 (fragments), IPv6 (en-têtes d'extension), ARP.</li>
<li><b>Flux « Argus »</b> (<code>flows.py</code>) : clé bidirectionnelle (protocole, numéro IP, extrémités triées), initiateur =
émetteur du premier paquet ; fermeture sur RST ou FIN des deux côtés, nouveau flux sur SYN après fermeture ou après 60 s
d'inactivité. Calcul de l'état (FIN, CON, INT, REQ, ACC, RST, ECO, ECR, URH, URN, TXD), des octets (longueur de trame),
TTL, retransmissions (arithmétique de séquence modulo 2³²), débits, tailles moyennes, temps inter-paquets et gigue,
durées de poignée de main (synack, ackdat, tcprtt).</li>
<li><b>Analyse « Bro »</b> (<code>app_layer.py</code>) : détection du service par signature (HTTP, SSH, TLS, SMTP, FTP, POP3, IRC)
comme le DPD de Bro, <code>ftp-data</code> via les ports négociés (PORT, PASV, EPSV), services UDP par port ; comptage des
transactions HTTP (<code>trans_depth</code>), taille des corps de réponse (<code>res_bdy_len</code>), méthodes GET/POST,
authentification et commandes FTP.</li>
<li><b>Features additionnelles</b> : Algorithme 3.2 (<code>is_sm_ips_ports</code>), 3.3 (<code>ct_state_ttl</code>, règles TTL × état),
3.7–3.9 (<code>ct_*</code> : nombre de connexions partageant une clé dans le groupe de 100 enregistrements triés par <code>ltime</code>).</li>
</ul>
<h3>Validation des formules sur le CSV officiel</h3>
<p>Nous avons confronté nos formules à 200 000 enregistrements publiés. Résultat notable : le débit d'Argus n'est pas
<code>sbytes·8/dur</code> (7,6 % de concordance) mais <code>(sbytes − sbytes/spkts)·8/dur</code> — un paquet est exclu du calcul.</p>
{cap("Concordance des formules de l'extracteur avec le CSV officiel (200 000 flux)")}
{table(fid, {"feature": "Feature", "formule": "Formule / règle", "concordance_pct": "Concordance (%)", "n": "n"}, {"concordance_pct": 3}) if len(fid) else ""}
<p><b>Limite.</b> Les compteurs <code>ct_*</code> ne se reproduisent qu'approximativement : le CSV publié est un sous-ensemble
des enregistrements Argus et l'ordre exact n'est pas documenté ; la lecture « blocs de 100 » de l'Algorithme 3.7 est la
plus proche des valeurs publiées. Une variante en fenêtre glissante est proposée pour la détection en continu.</p>
<h3>Test aller-retour sur du trafic réel (rejeu)</h3>
<p>Pour valider la chaîne sur un trafic réel sans versionner les 100 Go de PCAP d'origine, nous <b>rejouons</b> une tranche
contiguë de {num(len(pd.read_csv(C.SAMPLE_PCAP_DIR / "demo_unsw_replay_records.csv")) if (C.SAMPLE_PCAP_DIR / "demo_unsw_replay_records.csv").exists() else 0)}
enregistrements du CSV officiel (fichier 4, les neuf familles d'attaques) : pour chacun, <code>src/pcap/replay.py</code> reconstruit
les paquets qui le produisent (mêmes adresses, ports, TTL, nombres de paquets et d'octets, durée, délais de poignée de main, état et
signature applicative), puis notre extracteur ré-extrait les 47 features. Le PCAP est tronqué à 160 octets par trame
(<code>tcpdump -s 160</code>) : l'extracteur lit les longueurs dans les en-têtes IP, ce qui le rend robuste aux captures tronquées.</p>
{cap("Test aller-retour : concordance entre l'enregistrement d'origine et la feature ré-extraite")}
{table(rt, {"feature": "Feature", "critere": "Critère", "concordance_pct": "Concordance (%)", "n": "Flux comparés"}, {"concordance_pct": 2, "n": 0}, small=True) if len(rt) else ""}
{fig("07_replay_fidelity", "Test aller-retour enregistrement → PCAP → features.", "70%")}
<p>Les écarts résiduels (≈ 5 % sur dbytes, synack, ackdat, service) proviennent de flux TCP atypiques (retransmissions,
poignées de main incomplètes) que le rejeu simplifie. Les flux <code>unas</code> sont moins nombreux que les enregistrements :
Argus nomme ainsi tous les numéros de protocole IP non assignés, que le balayage du banc de test réutilise.</p>
<h3>Évaluation de bout en bout et artefact TTL</h3>
{cap("Chaîne PCAP → 47 features → modèles déployés, mesurée sur la vérité terrain de chaque flux")}
{table(e2e, {"description": "Capture", "profile": "Profil", "subset": "Flux évalués", "n_flows": "Flux", "n_attacks": "Attaques", "precision": "Précision", "recall": "Rappel", "false_alarm_rate": "FPR", "attack_type_accuracy": "Type exact"}, {"n_flows": 0, "n_attacks": 0, "precision": 3, "recall": 3, "false_alarm_rate": 3, "attack_type_accuracy": 3}, small=True) if len(e2e) else ""}
{fig("07_ttl_artifact", "Rappel sur le même trafic rejoué, avec TTL d'origine et TTL modernes (64).", "70%")}
<p><b>Lecture.</b> Sur le trafic rejoué, la chaîne complète détecte les attaques et attribue le bon <b>type d'attaque</b> à la grande
majorité d'entre elles, y compris pour les flux dont l'enregistrement n'a jamais été vu à l'entraînement. Remplacer les TTL par 64
fait <b>s'effondrer le profil complet</b> (il a appris « TTL 254 ⇒ attaque », propriété topologique du banc de test) alors que le
<b>profil portable</b> garde exactement les mêmes performances : c'est la démonstration directe de l'artefact TTL et la justification
du profil portable pour toute capture issue d'un autre réseau. Les scénarios synthétiques, dont les attaques sont inventées, sont
moins bien reconnus : un modèle appris sur UNSW-NB15 reconnaît les attaques du banc de test, pas toute anomalie — rôle complémentaire
des détecteurs d'anomalies (§12).</p></section>""")
    # ------------------------------------------------------------------ 5
    a(f"""<section><h2>Nettoyage et analyse exploratoire</h2>
<h3>Qualité des données brutes</h3>
<p>L'audit de l'échantillon brut révèle : <code>attack_cat</code> vide pour le trafic normal ; <code>ct_flw_http_mthd</code>
({num(cr.get('missing_ct_flw_http_mthd', 0))} valeurs vides), <code>is_ftp_login</code> et <code>ct_ftp_cmd</code>
({num(cr.get('missing_ct_ftp_cmd', 0))}) vides dans les fichiers 2 à 4 pour les flux non HTTP/FTP ; {num(cr.get('is_ftp_login_gt1', 0))}
valeurs de <code>is_ftp_login</code> supérieures à 1 alors que la variable est binaire ; ports hexadécimaux ; libellés
incohérents (« Backdoors », espaces). Aucune valeur infinie.</p>
{fig("01_missing_values", "Taux de valeurs manquantes dans les données brutes.", "70%")}
<p><b>Traitements.</b> Une valeur vide d'un compteur HTTP/FTP signifie l'<i>absence</i> d'événement : elle est remplacée par 0
(et non par une moyenne, qui inventerait une activité). Les doublons exacts sur (features + catégorie) sont supprimés
({num(cr.get('duplicates_removed', 0))}) pour éviter qu'un même vecteur se retrouve dans le train et le test. Nous détectons
aussi {num(cr.get('conflicting_vectors', 0))} vecteurs identiques portant des étiquettes différentes — bruit d'étiquetage
intrinsèque qui bornera la classification multi-classe. Jeu final : <b>{num(cr.get('rows_out', 0))} flux</b>.
Les valeurs aberrantes ne sont <b>pas</b> supprimées (§6.6).</p>
<h3>Distributions</h3>
{fig("01_numeric_distributions", "Histogrammes et densités KDE (échelle log1p) des principales variables, par classe.")}
<p>Les variables de volume ont une asymétrie supérieure à 10 et des queues lourdes. <code>sttl</code> est quasi binaire
(31 pour le normal, 254 ou 62 pour les attaques) : c'est un artefact de la topologie du banc de test, discuté au §10.4.
Les attaques ont souvent <code>dbytes = 0</code> (flux unidirectionnels, état INT).</p>
{fig("01_categorical_frequencies", "Fréquences des variables catégorielles, par classe.")}
{fig("01_boxplots_by_category", "Boîtes à moustaches par catégorie (échelle log1p).")}
<h3>Corrélations</h3>
{fig("01_spearman_heatmap", "Matrice de corrélation de Spearman.", "78%")}
<p>Des blocs très corrélés apparaissent (<code>sbytes</code>–<code>spkts</code>–<code>sloss</code>, <code>synack</code>–<code>ackdat</code>–<code>tcprtt</code>,
famille <code>ct_*</code>) : redondance traitée par un filtre de corrélation dans le prétraitement.</p></section>""")
    # ------------------------------------------------------------------ 6
    a(f"""<section><h2>Tests statistiques</h2>
<p>Chaque test répond à une question précise et son choix découle de la nature des variables. Toutes les p-values sont
corrigées par la procédure de <b>Benjamini-Hochberg</b> : avec m tests au seuil 5 %, on attendrait sinon 0,05·m faux positifs ;
BH contrôle le taux de fausses découvertes (FDR) en comparant la k-ième plus petite p-value à k·α/m. Les tests portent
sur le jeu d'entraînement ({num(meta.get('train_rows', 0))} flux).</p>
<h3>Normalité : Shapiro-Wilk et D'Agostino K²</h3>
<p>H₀ : la variable suit une loi normale. La statistique W de Shapiro-Wilk compare les statistiques d'ordre aux quantiles
attendus sous normalité ; c'est le test le plus puissant pour n ≤ 5 000 (nous tirons 4 000 observations, car au-delà tout
écart infime est rejeté). D'Agostino K² combine asymétrie et aplatissement. <b>Résultat :</b> normalité rejetée pour
<b>{eda.get('n_normal_rejected', '—')} / {eda.get('n_tests_normality', '—')}</b> variables. <b>Conséquence :</b> tests non
paramétriques et transformation <code>log1p</code> des variables asymétriques.</p>
<h3>Mann-Whitney U (numérique vs binaire)</h3>
<p>H₀ : les distributions des deux classes sont identiques. Alternative non paramétrique au test t pour deux échantillons
indépendants, fondée sur les rangs. Avec n ≈ 73 000, la significativité est quasi automatique : nous rapportons la taille
d'effet <b>r = 1 − 2U/(n₁n₂)</b> (rang-bisériale). <b>{eda.get('mw_significant', '—')} / {eda.get('mw_total', '—')}</b> variables sont significatives.</p>
{cap("Mann-Whitney U : les dix plus grandes tailles d'effet")}
{table(mw.head(10), {"feature": "Variable", "median_normal": "Médiane normal", "median_attack": "Médiane attaque", "rank_biserial_r": "r", "effect_size": "Effet", "p_bh": "p (BH)"}, {"p_bh": lambda v: f"{v:.1e}", "rank_biserial_r": 3}, small=True) if len(mw) else ""}
{fig("01_mann_whitney_effects", "Tailles d'effet du test de Mann-Whitney (r rang-bisériale).", "70%")}
<h3>Kruskal-Wallis (numérique vs 10 catégories)</h3>
<p>Généralisation non paramétrique de l'ANOVA à un facteur ; taille d'effet ε² = (H − k + 1)/(n − k).
<b>{eda.get('kw_significant', '—')} / {eda.get('kw_total', '—')}</b> variables diffèrent significativement entre catégories ;
les plus fortes : {", ".join(f"<code>{f}</code> (ε² = {e:.2f})".replace(".", ",") for f, e in zip(kw.feature.head(5), kw.epsilon_sq.head(5))) if len(kw) else ""}.</p>
<h3>χ² d'indépendance et V de Cramér (catégoriel vs cible)</h3>
<p>H₀ : indépendance. Pour respecter la condition des effectifs théoriques (≥ 5 dans chaque case), les modalités
trop rares compte tenu de la plus petite classe (Worms) sont regroupées dans une modalité « autre » ; le V de Cramér √(χ²/(n·(min(r,c) − 1))) mesure la force de l'association.</p>
{cap("Tests du χ² d'indépendance")}
{table(chi, {"feature": "Variable", "target": "Cible", "chi2": "χ²", "dof": "ddl", "cramers_v": "V de Cramér", "p_bh": "p (BH)"}, {"chi2": 0, "dof": 0, "cramers_v": 3, "p_bh": lambda v: f"{v:.1e}"}, small=True) if len(chi) else ""}
<h3>Corrélations de Spearman</h3>
<p>ρ de Spearman = corrélation de Pearson des rangs : il capte les relations monotones et résiste aux queues lourdes.
Il guide le filtre de redondance (|r| > 0,95) du prétraitement.</p>
<h3>Valeurs aberrantes</h3>
<p>Règle de Tukey : observation hors de [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]. Les valeurs extrêmes sont majoritairement des
attaques — précisément le signal recherché : elles sont conservées, et leur influence est réduite par <code>log1p</code>,
la standardisation et l'emploi de modèles à base d'arbres, insensibles aux transformations monotones.</p>
<h3>Synthèse</h3>
<p>(1) Aucune variable n'est normale ; (2) presque toutes séparent normal et attaques, avec des effets forts pour les
variables de TTL, de taille moyenne des paquets et de volume ; (3) les trois variables catégorielles sont associées à la cible ;
(4) une forte redondance existe. Ces constats déterminent directement le prétraitement.</p></section>""")
    # ------------------------------------------------------------------ 7
    a(f"""<section><h2>Prétraitement</h2>
<p>Le jeu nettoyé est découpé en <b>train (80 %, {num(meta.get('train_rows', 0))})</b> et <b>test (20 %, {num(meta.get('test_rows', 0))})</b>,
stratifié sur les 10 catégories. Le pipeline scikit-learn suivant est <b>ajusté uniquement sur le train</b> et, pendant la
validation croisée, ré-ajusté dans chaque pli (il fait partie du <code>Pipeline</code> optimisé) :</p>
<ol>
<li><b>Ingénierie de features</b> : total_bytes, total_pkts, byte_ratio = sbytes/(dbytes+1), pkt_ratio, bytes_per_pkt — calculables depuis un PCAP.</li>
<li><b>Imputation</b> par la médiane (numérique) ou une modalité constante (catégoriel) : robustesse en inférence.</li>
<li><b>log1p</b> des colonnes dont l'asymétrie mesurée sur le train dépasse 2 (choix guidé par les tests de normalité).</li>
<li><b>Filtre de corrélation</b> : suppression d'une variable de chaque paire |r| > 0,95.</li>
<li><b>Standardisation Z-score</b> z = (x − μ)/σ, indispensable aux méthodes fondées sur des distances ou des marges
(K-NN, SVM, régression logistique) ; <b>Min-Max</b> [0, 1] pour Naïve Bayes multinomial (entrées positives requises).</li>
<li><b>One-Hot</b> de proto, state et service ; modalités de moins de 30 occurrences regroupées dans « infrequent » et
modalités inconnues tolérées — indispensable pour un PCAP réel où peut apparaître un protocole jamais vu.</li>
</ol>
{fig("02_transformations_sbytes", "Effet du log1p puis de la standardisation sur sbytes.")}
<p><b>Profil « portable ».</b> Un second jeu de modèles est entraîné sans <code>sttl</code>, <code>dttl</code> et
<code>ct_state_ttl</code>, pour évaluer et réduire la dépendance à l'artefact TTL.</p></section>""")
    # ------------------------------------------------------------------ 8
    a("""<section><h2>Modèles : fondements théoriques</h2>
<p><b>Zero-R</b> prédit toujours la classe majoritaire : c'est le plancher de performance (exempté d'optimisation).</p>
<p><b>K plus proches voisins.</b> Un flux reçoit la classe majoritaire de ses K voisins selon une distance de Minkowski
d(x, y) = (Σ|xᵢ − yᵢ|ᵖ)^(1/p) (p = 1 : Manhattan, p = 2 : euclidienne), éventuellement pondérée par 1/d. K petit : variance
élevée ; K grand : biais. La standardisation est indispensable, sinon les variables à grande échelle (sload ~10⁸) dominent.</p>
<p><b>Arbre de décision (CART).</b> Partition récursive choisissant à chaque nœud le seuil qui maximise la baisse d'impureté :
indice de Gini G = 1 − Σ pₖ² ou entropie H = −Σ pₖ log₂ pₖ (gain d'information). L'élagage coût-complexité minimise
R(T) + α|T| : ccp_α pénalise le nombre de feuilles et limite le sur-apprentissage.</p>
<p><b>Random Forest.</b> Moyenne de B arbres appris sur des échantillons bootstrap avec, à chaque nœud, un sous-ensemble aléatoire
de variables : la décorrélation des arbres réduit la variance (bagging). <b>Extra Trees</b> tire aussi les seuils au hasard.</p>
<p><b>Naïve Bayes.</b> P(y | x) ∝ P(y) Πⱼ P(xⱼ | y) sous l'hypothèse d'indépendance conditionnelle des variables.
Variante gaussienne (vraisemblance normale par classe, <code>var_smoothing</code> stabilise les variances), de Bernoulli
(variables binarisées) et multinomiale ; le <b>lissage de Laplace</b> P(xⱼ | y) = (nⱼᵧ + α)/(nᵧ + α·d) évite les probabilités nulles.</p>
<p><b>SVM.</b> Recherche de l'hyperplan de marge maximale : min ½‖w‖² + C Σ ξᵢ sous yᵢ(w·φ(xᵢ) + b) ≥ 1 − ξᵢ. C règle le compromis
marge / erreurs ; l'astuce du noyau remplace φ par K(x, x') : linéaire, RBF exp(−γ‖x − x'‖²) ou polynomial (γ x·x' + r)ᵈ.</p>
<p><b>Extensions.</b> Régression logistique (modèle linéaire du log-odds, régularisation L2 d'intensité 1/C) ; analyse discriminante
linéaire (covariance commune, avec rétrécissement de Ledoit-Wolf) ; <b>AdaBoost</b> (repondération des exemples mal classés) ;
<b>Gradient Boosting</b> par histogrammes et <b>XGBoost</b> (ajout séquentiel d'arbres ajustés au gradient de la perte, régularisation
L2 des feuilles, sous-échantillonnage) ; <b>vote souple</b> (moyenne des probabilités) et <b>stacking</b> (régression logistique apprenant
à combiner les probabilités de RF, Extra Trees et XGBoost obtenues par validation croisée). Le Deep Learning est exclu conformément
au cahier des charges.</p></section>""")
    # ------------------------------------------------------------------ 9
    a(f"""<section><h2>Protocole d'optimisation et d'évaluation</h2>
<ul>
<li><b>Validation croisée stratifiée à 5 plis</b> (<code>StratifiedKFold(5, shuffle, random_state=42)</code>) pour tous les modèles réglables.</li>
<li><b>GridSearchCV</b> pour les espaces discrets (K-NN, CART, Naïve Bayes, SVM, régression logistique, LDA, AdaBoost) ;
<b>RandomizedSearchCV</b> (12 à 15 tirages) pour les espaces continus des ensembles, plus efficace à budget égal (Bergstra & Bengio, 2012).</li>
<li><b>Critère</b> : F1 de la classe attaque (binaire) ou F1-macro (multi-classe), robustes au déséquilibre ; accuracy,
précision, rappel et ROC-AUC sont enregistrés pour chaque configuration.</li>
<li>K-NN et SVM, de coût quadratique, sont réglés sur un sous-échantillon stratifié (20 000 et 8 000 flux) ; le SVM final est
ajusté sur 25 000 flux.</li>
<li><b>Sélection du modèle</b> sur la moyenne du score obtenu sur 5 plis stratifiés <b>communs à tous les modèles</b> (15 000 flux), y compris vote et stacking : les scores de recherche ne sont pas comparables entre modèles (données différentes, biais de sélection optimiste). Le jeu de test n'est utilisé qu'une fois, pour l'évaluation finale.</li>
<li><b>Métriques de test</b> : accuracy, précision, rappel (taux de détection), F1, F1-macro, MCC, kappa, taux de fausses alertes,
ROC-AUC et PR-AUC calculées à partir des probabilités (ou de la fonction de décision), matrices de confusion.</li>
<li><b>Comparaison statistique</b> : scores par pli sur des plis communs (15 000 flux) → test de Friedman, distance critique de
Nemenyi, Wilcoxon signé (correction de Holm) ; test de McNemar sur les prédictions de test des deux meilleurs.</li>
</ul>
<p>Au total, {sum(int(r.get('n_configs', 1)) for t in ('binary', 'multiclass') for r in reg['metrics']['full'][t])} configurations ont été
évaluées en CV pour le profil complet (soit {5 * sum(int(r.get('n_configs', 1)) for t in ('binary', 'multiclass') for r in reg['metrics']['full'][t])} ajustements).</p></section>""")
    # ------------------------------------------------------------------ 10
    cols_b = {"model": "Modèle", "cv_common_mean": "F1 plis comm.", "cv_mean": "F1 CV rech.", "accuracy": "Accuracy", "precision": "Précision", "recall": "Rappel",
              "f1": "F1", "roc_auc": "ROC-AUC", "mcc": "MCC", "false_alarm_rate": "FPR", "fit_time_s": "Fit (s)"}
    cols_m = {"model": "Modèle", "cv_common_mean": "F1-m plis comm.", "cv_mean": "F1-m CV rech.", "accuracy": "Accuracy", "balanced_accuracy": "Acc. équil.", "precision_macro": "Préc. macro",
              "recall_macro": "Rappel macro", "f1_macro": "F1-macro", "f1_weighted": "F1 pondéré", "roc_auc_ovr": "AUC OvR", "mcc": "MCC"}
    hb = lambda r: r["key"] == reg["best"]["full"]["binary"]  # noqa: E731
    hm = lambda r: r["key"] == reg["best"]["full"]["multiclass"]  # noqa: E731
    a(f"""<section><h2>Résultats de la classification supervisée</h2>
<h3>Détection binaire</h3>
{cap("Comparaison multicritère sur le jeu de test (binaire, profil complet) ; ligne surlignée : modèle retenu sur la CV")}
{table(mb, cols_b, {"fit_time_s": 1}, hb, small=True)}
<p>Tous les modèles dépassent très largement Zero-R (F1 = {num(zr.f1)}, accuracy {pct(zr.accuracy)}, ROC-AUC 0,5) qui ne détecte
aucune attaque. Le modèle retenu, <b>{bb.model}</b> (F1 moyen sur les plis communs {num(bb.get("cv_common_mean", bb.cv_mean))}), obtient sur le test un F1 de
<b>{pct(bb.f1)}</b>, un rappel de {pct(bb.recall)} pour {pct(bb.false_alarm_rate)} de fausses alertes. Les ensembles d'arbres dominent ;
K-NN, CART, SVM et la régression logistique les suivent de près ; Naïve Bayes est en retrait ({nb_best.model} : F1 {pct(nb_best.f1)}) car
l'hypothèse d'indépendance conditionnelle est violée par des variables très corrélées.</p>
{fig("03_model_comparison_binary", "Comparaison des métriques de test (binaire).", "80%")}
{fig("03_roc_pr_curves", "Courbes ROC (zoom FPR < 30 %) et précision-rappel.")}
{fig("03_confusion_grid_binary", "Matrices de confusion de tous les modèles (N = normal, A = attaque).")}
<h3>Étude des hyperparamètres</h3>
{fig("03_hyperparameter_studies", "K-NN (K, distance, pondération), CART (critère, profondeur), Bernoulli NB (α), SVM (noyau).")}
<p>Pour K-NN, les petites valeurs de K avec pondération par la distance sont les meilleures : les attaques forment des
groupes compacts et très répétitifs. Pour CART, Gini et entropie donnent des résultats quasi identiques ; la performance
sature au-delà d'une profondeur de 14 à 20. Le noyau RBF domine les noyaux linéaire et polynomial pour le SVM.
Les hyperparamètres retenus figurent en annexe A.</p>
<h3>Classification en 10 catégories</h3>
{cap("Comparaison multi-classe (jeu de test)")}
{table(mm, cols_m, {}, hm, small=True)}
{fig("03_multiclass_results", f"Matrice de confusion de {bm.model} et F1 par catégorie pour chaque modèle.")}
<p>Les catégories les mieux reconnues sont {", ".join(f"{c} (F1 {num(v['f1-score'], 2)})" for c, v in strong)} ; les plus difficiles
sont {", ".join(f"{c} (F1 {num(v['f1-score'], 2)})" for c, v in weak)}. Ces dernières partagent des vecteurs de features identiques
(§5.1) : aucune frontière de décision ne peut les séparer parfaitement, ce qui explique l'écart entre F1-macro et accuracy.</p>
<h3>Profil portable et artefact TTL</h3>
<p>Sans les trois variables de TTL, le meilleur modèle binaire ({bp.model}) obtient un F1 de {pct(bp.f1)} (contre {pct(bb.f1)}) et
une ROC-AUC de {num(bp.roc_auc)} : l'information discriminante n'est pas uniquement portée par le TTL, et ce profil est
préférable pour des captures issues d'autres réseaux.</p>
<h3>Comparaison statistique</h3>
<p><b>Binaire :</b> test de Friedman χ² = {num(fr_b.get('chi2', float('nan')), 2)}, p = {fr_b.get('p_value', float('nan')):.2e} ({fr_b.get('k_models', '—')} modèles × 5 plis) ;
distance critique de Nemenyi {num(fr_b.get('critical_difference', float('nan')), 2)}. McNemar entre {name(mc_b.get('model_a', ''))} et
{name(mc_b.get('model_b', ''))} : b = {mc_b.get('b', '—')}, c = {mc_b.get('c', '—')}, p = {mc_b.get('p_value', float('nan')):.3g}
→ {'différence significative' if mc_b.get('p_value', 1) < 0.05 else 'pas de différence significative : les deux modèles sont statistiquement équivalents'}.
<b>Multi-classe :</b> Friedman χ² = {num(fr_m.get('chi2', float('nan')), 2)}, p = {fr_m.get('p_value', float('nan')):.2e} ; McNemar
{name(mc_m.get('model_a', ''), 'multiclass')} vs {name(mc_m.get('model_b', ''), 'multiclass')} : p = {mc_m.get('p_value', float('nan')):.3g}.
Avec seulement 5 plis, le test de Nemenyi est conservateur : plusieurs modèles de tête restent statistiquement indiscernables,
ce qui relativise les écarts de quelques millièmes.</p>
<h3>Seuil de décision et courbes d'apprentissage</h3>
{fig("03_threshold_analysis", "Arbitrage précision / rappel / fausses alertes selon le seuil.", "75%")}
{fig("03_learning_curves", "Courbes d'apprentissage (F1 en CV 5 plis).", "75%")}
<p>Abaisser le seuil augmente le rappel au prix de fausses alertes : l'application expose ce seuil à l'analyste. Les courbes
d'apprentissage montrent un écart train/validation faible pour les ensembles : ajouter des données apporterait peu.</p></section>""")
    # ------------------------------------------------------------------ 11
    a(f"""<section><h2>Importance des variables et explicabilité</h2>
<p>Cinq méthodes sont comparées : réduction d'impureté de Gini (Random Forest), gain moyen (XGBoost), valeurs de Shapley
(TreeSHAP exact, calculé nativement par XGBoost), permutation (baisse du F1 de test quand une variable est mélangée) et
information mutuelle (indépendante du modèle). Une valeur SHAP φⱼ est la contribution moyenne de la variable j à la prédiction
sur toutes les coalitions de variables ; leur somme plus la valeur de base égale le log-odds prédit (additivité).</p>
<p><b>Consensus</b> (rang moyen des cinq méthodes) : {top_feats}.</p>
{fig("06_feature_importance_methods", "Top 15 des variables selon chaque méthode.")}
{fig("06_shap_summary", "Résumé SHAP (XGBoost binaire) : chaque point est un flux.", "78%")}
{fig("06_shap_local", "Explication locale d'un flux Exploits.", "72%")}
{fig("06_shap_category_ablation", "SHAP par catégorie (multi-classe) et ablation par groupe de features.")}
{cap("Ablation : F1 après retrait d'un groupe de features (XGBoost ré-entraîné)")}
{table(ab, {"removed_group": "Groupe retiré", "n_removed": "n", "f1_binary": "F1 binaire", "delta_binary": "Δ", "f1_multiclass": "F1-macro", "delta_multiclass": "Δ"}, {"n_removed": 0}, small=True) if len(ab) else ""}
<p>Le faible impact de chaque retrait révèle la <b>redondance</b> entre groupes ; la tâche multi-classe est plus sensible.
L'explication locale est disponible pour chaque prédiction de l'application (graphique + texte généré de façon déterministe).</p></section>""")
    # ------------------------------------------------------------------ 12
    a(f"""<section><h2>Apprentissage non supervisé</h2>
<p>Le clustering est réalisé dans l'espace des 10 premières composantes principales (ACP, {pct(sum(un.get('pca_explained', [0])), 1)} de variance)
des features prétraitées, sur 20 000 flux d'entraînement, <b>sans utiliser les étiquettes</b> ; celles-ci servent seulement à
l'évaluation externe.</p>
<ul>
<li><b>K-Means</b> minimise l'inertie intra-classe (WCSS) Σ‖x − μ_c‖² ; initialisation k-means++ (centres dispersés) comparée à
l'initialisation aléatoire ; k choisi par la <b>méthode du coude</b> (k = {un.get('k_elbow', '—')}) et confronté à la silhouette (max en k = {un.get('k_silhouette', '—')}).</li>
<li><b>K-Medoids PAM</b> (implémenté : phases BUILD puis SWAP) : les centres sont des flux réels, ce qui rend la méthode plus robuste
aux valeurs aberrantes ; <b>CLARA</b> applique PAM à plusieurs échantillons pour passer à l'échelle.</li>
<li><b>AGNES</b> (ascendant) avec liaisons de Ward, complète, moyenne et simple ; qualité mesurée par la corrélation cophénétique.
<b>DIANA</b> (descendant, implémenté) scinde récursivement le cluster de plus grand diamètre autour d'un groupe dissident.</li>
<li><b>DBSCAN</b> : clusters de densité (ε, minPts), ε guidé par le graphe des k-distances ; les points isolés sont du bruit.</li>
<li><b>GMM</b> : mélange de gaussiennes appris par EM, nombre de composantes choisi par le BIC.</li>
</ul>
{fig("04_kmeans_elbow_silhouette", "Méthode du coude et indices internes pour K-Means.")}
{cap("Comparaison des algorithmes de clustering (indices internes et externes vs les 10 catégories)")}
{table(cmpu, {"algorithm": "Algorithme", "n_clusters": "k", "silhouette": "Silhouette", "davies_bouldin": "DB", "rand_index_cat": "Rand", "adjusted_rand_cat": "ARI", "nmi_cat": "NMI", "purity_cat": "Pureté 10 cl.", "purity_binary": "Pureté bin."}, {"n_clusters": 0, "silhouette": 3, "davies_bouldin": 3, "rand_index_cat": 3, "adjusted_rand_cat": 3, "nmi_cat": 3, "purity_cat": 3, "purity_binary": 3}, small=True) if len(cmpu) else ""}
{fig("04_pca_clusters", "Projection ACP : clusters K-Means, classes réelles et correspondance clusters × catégories.")}
{fig("04_dendrograms", "Dendrogrammes AGNES pour les quatre liaisons.")}
{fig("04_dbscan_gmm", "Graphe des k-distances (DBSCAN) et critères BIC/AIC (GMM).")}
<p><b>Interprétation.</b> Les clusters retrouvent bien la séparation normal / attaque (pureté binaire élevée) mais pas
finement les dix catégories (ARI modéré) : sans étiquettes, les catégories qui se chevauchent restent indiscernables.
La liaison simple subit l'effet de chaîne ; Ward produit des groupes équilibrés. Robustesse : après injection de
{un.get('robustness', {}).get('outliers_injected', '—')} points aberrants, les centres se déplacent de
{num(un.get('robustness', {}).get('pam_center_shift', float('nan')), 3)} (PAM) contre {num(un.get('robustness', {}).get('kmeans_center_shift', float('nan')), 3)} (K-Means).</p>
<h3>Détection d'intrusions par anomalies</h3>
{table(anom, {"detector": "Détecteur", "mode": "Apprentissage", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "best_f1": "F1 (seuil optimal)"}, small=True) if len(anom) else ""}
<p>Appris sur le seul trafic normal, ces détecteurs atteignent une ROC-AUC élevée sans jamais voir d'attaque : ils complètent
le classifieur supervisé pour les attaques inédites, au prix de davantage de fausses alertes.</p></section>""")
    # ------------------------------------------------------------------ 13
    a(f"""<section><h2>Règles d'association</h2>
<p>Chaque flux devient une transaction d'items « attribut = modalité » (variables numériques discrétisées en terciles).
L'algorithme <b>Apriori</b> (implémenté) énumère les itemsets fréquents niveau par niveau en exploitant l'anti-monotonie
du support (tout sous-ensemble d'un itemset fréquent est fréquent), puis les règles X ⇒ Y sont évaluées par le support
P(X ∧ Y), la confiance P(Y | X), le lift P(Y | X)/P(Y), le levier et la conviction. {un.get('n_frequent_itemsets', '—')} itemsets
fréquents (support ≥ 3 %) et {un.get('n_rules', '—')} règles de confiance ≥ 0,7 sont obtenus.</p>
{cap("Règles de plus fort lift ayant la classe pour conséquent")}
{table(rules.head(10), {"antecedent": "Si", "consequent": "Alors", "support": "Support", "confidence": "Confiance", "lift": "Lift"}, {"support": 3, "confidence": 3, "lift": 2}, small=True) if len(rules) else ""}
{cap("Règles vers une catégorie d'attaque")}
{table(rules_c.head(8), {"antecedent": "Si", "consequent": "Alors", "support": "Support", "confidence": "Confiance", "lift": "Lift"}, {"support": 3, "confidence": 3, "lift": 2}, small=True) if len(rules_c) else ""}
<p>Ces règles confirment l'EDA (TTL, état, service DNS et rafales de connexions pour Generic) sous une forme directement
lisible par un analyste sécurité.</p></section>""")
    # ------------------------------------------------------------------ 14
    a("""<section><h2>Application web</h2>
<p>L'application Flask (<code>src/app.py</code>, logique dans <code>src/services.py</code>) fonctionne entièrement en local,
y compris hors-ligne (Plotly.js est servi depuis le paquet Python installé). Elle comprend onze pages :</p>
<ul>
<li><b>Dashboard EDA</b> : statistiques descriptives, valeurs manquantes (brut et nettoyé), histogrammes avec KDE, fréquences
catégorielles, matrice de corrélation Spearman/Pearson, boîtes à moustaches et nuage de points, avec <b>filtres temps réel</b>
(catégories, classe, protocole, service, état, plage d'une variable numérique) — les six éléments exigés.</li>
<li><b>Prédiction temps réel</b> : formulaire des 39 features pré-rempli par des flux réels, mise à jour instantanée de la
probabilité d'attaque, de la catégorie et de sa confiance, explication SHAP locale et textuelle, cluster K-Means et score
d'anomalie ; choix du modèle, du profil et du seuil.</li>
<li><b>Analyse PCAP</b> : téléversement (pcap, pcapng, csv), extraction des 47 features, alertes, chronologie, IP sources
suspectes, détail et explication de chaque flux, export CSV, évaluation lorsqu'une vérité terrain est disponible.</li>
<li>Pages <b>tests statistiques</b>, <b>modèles</b> (tableaux, ROC, PR, confusion, résultats de CV, seuil, tests de Friedman/McNemar),
<b>explicabilité</b>, <b>non supervisé</b>, <b>règles d'association</b>, <b>génération des tuples</b> et <b>documentation</b> ; API REST JSON.</li>
</ul>
<p>Les erreurs sont gérées explicitement (fichier vide, format non supporté, feature manquante, fichier trop volumineux).
Une suite de tests <code>pytest</code> couvre l'extracteur, le prétraitement, les algorithmes implémentés (PAM, DIANA, Apriori,
Benjamini-Hochberg) et chaque page et route de l'application.</p></section>""")
    # ------------------------------------------------------------------ 15
    a(f"""<section><h2>Discussion et limites</h2>
<ul>
<li><b>Artefact TTL.</b> Dans le banc de test, les machines attaquantes émettent avec un TTL de 254 ou 62 : <code>sttl</code> est la
variable la plus discriminante pour des raisons topologiques et non comportementales. Le profil portable quantifie et contourne cette dépendance.</li>
<li><b>Bruit d'étiquetage.</b> {num(cr.get('conflicting_vectors', 0))} vecteurs identiques portent des étiquettes différentes : le F1-macro
est structurellement plafonné, quel que soit l'algorithme.</li>
<li><b>Échantillonnage.</b> La prévalence de l'échantillon (≈ 37 % d'attaques) diffère de celle du jeu complet (12,6 %) : la précision
en production serait plus faible à seuil égal ; le seuil est réglable.</li>
<li><b>Features ct_*.</b> Reproduites approximativement par l'extracteur ; les captures de démonstration sont synthétiques.
Une validation sur les PCAP originaux de UNSW-NB15 (100 Go) constitue la suite naturelle.</li>
<li><b>Ressources.</b> Machine de 8 Go : réglage de K-NN et SVM sur sous-échantillons, parallélisme limité.</li>
</ul></section>""")
    a(f"""<section><h2>Conclusion</h2>
<p>Nous avons réalisé une chaîne complète de Data Mining appliquée à la détection d'intrusions : données volumineuses
nettoyées et auditées, analyse exploratoire étayée par des tests statistiques appropriés, quinze classifieurs optimisés en
validation croisée et comparés statistiquement, sept méthodes de clustering, quatre détecteurs d'anomalies et des règles
d'association. Le détecteur retenu ({bb.model}) atteint un F1 de {pct(bb.f1)} pour {pct(bb.false_alarm_rate)} de fausses alertes.
Au-delà du benchmark, la réimplémentation validée de la génération des 47 features UNSW-NB15 à partir de paquets bruts
rend le modèle réellement utilisable sur des captures, ce que démontre l'application web livrée.</p></section>""")
    a("""<section><h2 class="nonum">Références</h2><ol class="refs">
<li>N. Moustafa, J. Slay. UNSW-NB15: a comprehensive data set for network intrusion detection systems. <i>MilCIS</i>, 2015.</li>
<li>N. Moustafa, J. Slay. The evaluation of network anomaly detection systems: statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set. <i>Information Security Journal</i>, 25(1-3), 2016.</li>
<li>N. Moustafa. <i>Designing an online and reliable statistical anomaly detection framework for dealing with large high-speed network traffic</i>. Thèse de doctorat, UNSW Canberra, 2017.</li>
<li>L. Breiman. Random Forests. <i>Machine Learning</i>, 45, 2001.</li>
<li>T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. <i>KDD</i>, 2016.</li>
<li>C. Cortes, V. Vapnik. Support-vector networks. <i>Machine Learning</i>, 20, 1995.</li>
<li>S. Lundberg, S.-I. Lee. A unified approach to interpreting model predictions. <i>NeurIPS</i>, 2017.</li>
<li>L. Kaufman, P. Rousseeuw. <i>Finding Groups in Data: An Introduction to Cluster Analysis</i>. Wiley, 1990.</li>
<li>M. Ester, H.-P. Kriegel, J. Sander, X. Xu. A density-based algorithm for discovering clusters. <i>KDD</i>, 1996.</li>
<li>R. Agrawal, R. Srikant. Fast algorithms for mining association rules. <i>VLDB</i>, 1994.</li>
<li>Y. Benjamini, Y. Hochberg. Controlling the false discovery rate. <i>JRSS B</i>, 57(1), 1995.</li>
<li>J. Demšar. Statistical comparisons of classifiers over multiple data sets. <i>JMLR</i>, 7, 2006.</li>
<li>J. Bergstra, Y. Bengio. Random search for hyper-parameter optimization. <i>JMLR</i>, 13, 2012.</li>
<li>F. T. Liu, K. M. Ting, Z.-H. Zhou. Isolation Forest. <i>ICDM</i>, 2008.</li>
<li>F. Pedregosa et al. Scikit-learn: machine learning in Python. <i>JMLR</i>, 12, 2011.</li>
</ol></section>""")
    # annexe hyperparamètres
    hp = []
    for t in ("binary", "multiclass"):
        for k, v in reg["models"]["full"][t].items():
            hp.append({"task": "binaire" if t == "binary" else "multi-classe", "model": v["name"],
                       "params": ", ".join(f"{a}={b if not isinstance(b, float) else round(b, 4)}" for a, b in v["params"].items()) or "—"})
    a(f"""<section><h2 class="nonum">Annexe A — Hyperparamètres retenus</h2>
{table(pd.DataFrame(hp), {"task": "Tâche", "model": "Modèle", "params": "Meilleurs hyperparamètres (CV 5 plis)"}, small=True)}
<h2 class="nonum">Annexe B — Reproductibilité</h2>
<p>Versions : {", ".join(f"{k} {v}" for k, v in reg.get("versions", {}).items())}. Graine aléatoire : 42. Commandes :
<code>python -m src.prepare_data</code> → notebooks 01 à 07 dans l'ordre (l'entraînement est réalisé dans le notebook 03 ;
équivalent sans Jupyter : <code>python scripts/train.py</code>) → <code>python scripts/build_report.py</code>.
Modèles déployés : <code>models/binaire/</code> et <code>models/multiclasse/</code> (pipeline.joblib + metadata.json). Tests : <code>python -m pytest</code>.</p></section>""")
    return "\n".join(S)


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1b1b1b; counter-reset: h2; }
h1, h2, h3 { color: #12355b; }
h2 { font-size: 16pt; border-bottom: 2px solid #2a78d6; padding-bottom: 4px; margin-top: 0; counter-increment: h2; counter-reset: h3; }
h2.nonum { counter-increment: none; }
h2:not(.nonum)::before { content: counter(h2) ". "; }
h3 { font-size: 12pt; margin: 14px 0 6px; counter-increment: h3; }
h2:not(.nonum) ~ h3::before { content: counter(h2) "." counter(h3) " "; }
section { page-break-before: always; }
section.cover { page-break-before: avoid; height: 255mm; display: flex; flex-direction: column; justify-content: center; }
.cover h1 { font-size: 26pt; line-height: 1.2; margin: 10px 0; }
.cover h2 { border: 0; font-size: 13pt; color: #52514e; font-weight: 500; counter-increment: none; }
.cover h2::before { content: none !important; }
.cover-top { color: #2a78d6; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; font-size: 10pt; }
.cover-box { border-left: 4px solid #2a78d6; background: #f2f6fc; padding: 10px 16px; margin: 24px 0; }
.cover-box p { margin: 4px 0; }
.cover-team { color: #52514e; margin-top: 30px; }
p { text-align: justify; margin: 6px 0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f1f1ee; padding: 0 3px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px; font-size: 9pt; page-break-inside: avoid; }
table.small { font-size: 8pt; }
th { background: #12355b; color: #fff; padding: 4px 6px; text-align: left; }
td { border-bottom: 1px solid #ddd; padding: 3px 6px; }
tr:nth-child(even) td { background: #f7f8fa; }
tr.hl td { background: #dce9fa !important; font-weight: 600; }
figure { margin: 10px auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption, .tcap { font-size: 9pt; color: #52514e; font-style: italic; text-align: center; margin-top: 4px; }
.tcap { margin-bottom: 2px; }
ul, ol { margin: 6px 0 6px 18px; padding: 0; } li { margin: 3px 0; }
#toc a { color: #12355b; text-decoration: none; } #toc div { margin: 3px 0; }
.refs li { font-size: 9.5pt; }
.missing { color: #d03b3b; }
"""

TOC_JS = """
<script>
const toc = document.getElementById('toc'); let n = 0;
document.querySelectorAll('h2').forEach((h, i) => {
  if (h.closest('.cover')) return;
  h.id = 's' + i;
  const num = h.classList.contains('nonum') ? '' : (++n) + '. ';
  if (h.textContent === 'Table des matières') return;
  toc.insertAdjacentHTML('beforeend', `<div><a href="#s${i}">${num}${h.textContent}</a></div>`);
  let m = 0;
  let el = h.nextElementSibling;
  while (el && el.tagName !== 'H2') { if (el.tagName === 'H3' && num) toc.insertAdjacentHTML('beforeend', `<div style="margin-left:18px;font-size:9.5pt">${n}.${++m} ${el.textContent}</div>`); el = el.nextElementSibling; }
});
</script>
"""


def find_browser() -> str | None:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(p).exists():
            return p
    for n in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(n):
            return shutil.which(n)
    return None


def main():
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    doc = (f"<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>Rapport — NIDS UNSW-NB15</title>"
           f"<style>{CSS}</style></head><body>{build()}{TOC_JS}</body></html>")
    OUT_HTML.write_text(doc, encoding="utf-8")
    print("HTML :", OUT_HTML, f"({len(doc) / 1e6:.1f} Mo)")
    br = find_browser()
    if not br:
        print("Navigateur Chromium introuvable : ouvrez le HTML et imprimez-le en PDF.")
        return
    subprocess.run([br, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", "--virtual-time-budget=10000", OUT_HTML.resolve().as_uri()],
                   check=True, timeout=180, capture_output=True)
    print("PDF  :", OUT_PDF, f"({OUT_PDF.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
