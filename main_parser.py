import pandas as pd
import os
from src.capture.parser import lire_pcap
from src.capture.flow_builder import grouper_en_flux

def pipeline_pcap_to_csv(chemin_pcap, chemin_csv_sortie):
    if not os.path.exists(chemin_pcap):
        print(f"❌ Erreur : Le fichier de capture réseau '{chemin_pcap}' est introuvable.")
        print("Veuillez vérifier qu'un fichier valide est bien placé dans le dossier 'data' et nommé 'sample.pcap'.")
        return

    # 1. Lecture du fichier de paquets
    liste_paquets = lire_pcap(chemin_pcap)
    if not liste_paquets:
        return
    
    # 2. Regroupement par flux de communication
    liste_flux = grouper_en_flux(liste_paquets)
    
    # 3. Structuration en tableau et sauvegarde au format attendu (CSV)
    df = pd.DataFrame(liste_flux)
    if not df.empty:
        df['network_bytes'] = df['sbytes'] + df['dbytes']
        if 'start_time' in df.columns:
            df = df.drop(columns=['start_time'])
            
    df.to_csv(chemin_csv_sortie, index=False)
    print(f"✅ Opération réussie ! Le fichier CSV final est disponible ici : {chemin_csv_sortie}")

if __name__ == "__main__":
    fichier_entree = "data/sample.pcap" 
    fichier_sortie = "data/trafic_client.csv"
    
    pipeline_pcap_to_csv(fichier_entree, fichier_sortie)
