from pathlib import Path

import pandas as pd

from src.capture.flow_builder import grouper_en_flux
from src.capture.parser import lire_pcap


def pipeline_pcap_to_csv(chemin_pcap, chemin_csv_sortie):
    pcap_path = Path(chemin_pcap)
    if not pcap_path.is_file():
        raise FileNotFoundError(f"Fichier de capture introuvable : {pcap_path}")

    packets = lire_pcap(pcap_path)
    if not packets:
        raise ValueError(f"Aucun paquet IP exploitable dans {pcap_path}")

    flows = grouper_en_flux(packets)
    if not flows:
        raise ValueError(f"Aucun flux exploitable dans {pcap_path}")

    output_path = Path(chemin_csv_sortie)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame.from_records(flows)
    dataframe.to_csv(output_path, index=False)

    print(f"CSV de métriques observées écrit : {output_path}")
    print(
        "Attention : les lignes sont marquées model_ready=False. "
        "L'extracteur ne reproduit pas le schéma UNSW-NB15 du modèle."
    )
    return dataframe


if __name__ == "__main__":
    pipeline_pcap_to_csv("data/sample.pcap", "data/trafic_client.csv")
