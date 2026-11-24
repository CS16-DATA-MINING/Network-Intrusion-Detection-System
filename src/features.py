"""Schéma des features UNSW-NB15 — source unique de vérité.

La liste ``FEATURES`` est utilisée à la fois pour l'entraînement (CSV UNSW-NB15)
et pour l'inférence sur un PCAP (``src.pcap``) : aucune colonne que l'extracteur
ne sait pas produire n'est utilisée par un modèle.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Schéma brut UNSW-NB15 (49 colonnes, ordre officiel du fichier features.csv)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
    "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

# Groupes de features tels que décrits dans la thèse (Tables 3.10 à 3.14)
FEATURE_GROUPS = {
    "flow": ["srcip", "sport", "dstip", "dsport", "proto"],
    "basic": ["state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
              "dloss", "service", "sload", "dload", "spkts", "dpkts"],
    "content": ["swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
                "trans_depth", "res_bdy_len"],
    "time": ["sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt",
             "synack", "ackdat"],
    "additional_general": ["is_sm_ips_ports", "ct_state_ttl",
                           "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"],
    "additional_connection": ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
                              "ct_src_ltm", "ct_src_dport_ltm",
                              "ct_dst_sport_ltm", "ct_dst_src_ltm"],
}

# Colonnes d'identification : affichées dans l'application mais JAMAIS utilisées
# comme features (fuite d'information : adresses/horodatages du banc de test).
ID_COLUMNS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]
# Numéros de séquence initiaux TCP : aléatoires par construction (RFC 6528),
# donc non généralisables -> exclus.
EXCLUDED_NOISE = ["stcpb", "dtcpb"]

TARGET_BINARY = "label"
TARGET_MULTI = "attack_cat"

CATEGORICAL_FEATURES = ["proto", "state", "service"]
NUMERIC_FEATURES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "sload",
    "dload", "spkts", "dpkts", "swin", "dwin", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit", "sintpkt", "dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
]
# Toutes les features d'entrée (avant ingénierie), dans un ordre fixe.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features dépendantes du TTL : artefact connu du banc de test UNSW-NB15
# (les hôtes attaquants ont des TTL 254/62). Le profil « portable » les retire
# pour mieux généraliser sur un PCAP capturé ailleurs.
TTL_FEATURES = ["sttl", "dttl", "ct_state_ttl"]

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms",
]

FEATURE_DESCRIPTIONS = {
    "srcip": "Adresse IP source", "sport": "Port source",
    "dstip": "Adresse IP destination", "dsport": "Port destination",
    "proto": "Protocole de transport (tcp, udp, icmp, ...)",
    "state": "État de la connexion (FIN, CON, INT, REQ, RST, ...)",
    "dur": "Durée totale du flux (s)",
    "sbytes": "Octets source → destination",
    "dbytes": "Octets destination → source",
    "sttl": "TTL source → destination", "dttl": "TTL destination → source",
    "sloss": "Paquets source retransmis/perdus",
    "dloss": "Paquets destination retransmis/perdus",
    "service": "Service applicatif (http, ftp, dns, smtp, ssh, ... ou -)",
    "sload": "Débit source (bits/s)", "dload": "Débit destination (bits/s)",
    "spkts": "Nombre de paquets source → destination",
    "dpkts": "Nombre de paquets destination → source",
    "swin": "Fenêtre TCP annoncée par la source",
    "dwin": "Fenêtre TCP annoncée par la destination",
    "stcpb": "Numéro de séquence TCP initial source",
    "dtcpb": "Numéro de séquence TCP initial destination",
    "smeansz": "Taille moyenne des paquets source (octets)",
    "dmeansz": "Taille moyenne des paquets destination (octets)",
    "trans_depth": "Profondeur de pipeline des transactions HTTP",
    "res_bdy_len": "Taille du contenu des réponses HTTP (octets)",
    "sjit": "Gigue source (ms)", "djit": "Gigue destination (ms)",
    "stime": "Horodatage de début", "ltime": "Horodatage de fin",
    "sintpkt": "Temps inter-paquets moyen source (ms)",
    "dintpkt": "Temps inter-paquets moyen destination (ms)",
    "tcprtt": "RTT d'établissement TCP (synack + ackdat)",
    "synack": "Délai SYN → SYN/ACK (s)", "ackdat": "Délai SYN/ACK → ACK (s)",
    "is_sm_ips_ports": "1 si IP et ports source = destination (attaque land)",
    "ct_state_ttl": "Code état × plages de TTL (Algorithme 3.3)",
    "ct_flw_http_mthd": "Nombre de méthodes HTTP (GET/POST) dans le flux",
    "is_ftp_login": "1 si session FTP authentifiée (USER + PASS)",
    "ct_ftp_cmd": "Nombre de commandes FTP dans le flux",
    "ct_srv_src": "Connexions même service + même IP source (fenêtre 100)",
    "ct_srv_dst": "Connexions même service + même IP destination (fenêtre 100)",
    "ct_dst_ltm": "Connexions même IP destination (fenêtre 100)",
    "ct_src_ltm": "Connexions même IP source (fenêtre 100)",
    "ct_src_dport_ltm": "Connexions même IP source + port destination (fenêtre 100)",
    "ct_dst_sport_ltm": "Connexions même IP destination + port source (fenêtre 100)",
    "ct_dst_src_ltm": "Connexions même couple IP source/destination (fenêtre 100)",
    "attack_cat": "Catégorie d'attaque (Normal + 9 familles)",
    "label": "0 = trafic normal, 1 = attaque",
}
