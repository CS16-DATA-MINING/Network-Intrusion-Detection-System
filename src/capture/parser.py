from scapy.all import PcapReader, IP, TCP, UDP

def lire_pcap(chemin_fichier):
    print(f"Lecture optimisée (Streaming) du fichier PCAP : {chemin_fichier}...")
    liste_paquets_extraits = []
    compteur = 0

    try:
        # PcapReader ouvre le fichier sans le charger entièrement en mémoire
        with PcapReader(chemin_fichier) as pcap_reader:
            for pkt in pcap_reader:
                if IP in pkt:
                    ip_src = pkt[IP].src
                    ip_dst = pkt[IP].dst
                    proto = pkt[IP].proto
                    taille = len(pkt)
                    timestamp = float(pkt.time)
                    
                    port_src, port_dst = 0, 0

                    if TCP in pkt:
                        port_src = pkt[TCP].sport
                        port_dst = pkt[TCP].dport
                        proto_nom = "tcp"
                    elif UDP in pkt:
                        port_src = pkt[UDP].sport
                        port_dst = pkt[UDP].dport
                        proto_nom = "udp"
                    else:
                        proto_nom = str(proto)

                    liste_paquets_extraits.append({
                        'ip_src': ip_src, 'port_src': port_src,
                        'ip_dst': ip_dst, 'port_dst': port_dst,
                        'proto': proto_nom, 'taille': taille, 'timestamp': timestamp
                    })
                    
                    compteur += 1
                    # Petit affichage tous les 100 000 paquets pour suivre l'avancement
                    if compteur % 100000 == 0:
                        print(f"➔ {compteur} paquets analysés...")
                        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier volumineux : {e}")
        return []
            
    print(f"➔ Analyse terminée. {len(liste_paquets_extraits)} paquets IP extraits au total.")
    return liste_paquets_extraits
