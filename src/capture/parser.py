from scapy.all import rdpcap, IP, TCP, UDP

def lire_pcap(chemin_fichier):
    print(f"Lecture du fichier PCAP : {chemin_fichier}...")
    try:
        paquets = rdpcap(chemin_fichier)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier : {e}")
        return []

    liste_paquets_extraits = []

    for pkt in paquets:
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
            
    print(f"➔ {len(liste_paquets_extraits)} paquets IP extraits avec succès.")
    return liste_paquets_extraits
