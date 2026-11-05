from scapy.all import wrpcap, IP, TCP, UDP
import os

def creer_faux_pcap():
    print("Fabrication d'un fichier PCAP de test local...")
    
    # On s'assure que le dossier data existe
    os.makedirs("data", exist_ok=True)
    
    # On simule quelques paquets réseau (du trafic normal et suspect)
    paquets = [
        IP(src="192.168.1.10", dst="10.0.0.5")/TCP(sport=443, dport=50210),
        IP(src="10.0.0.5", dst="192.168.1.10")/TCP(sport=50210, dport=443),
        IP(src="192.168.1.22", dst="8.8.8.8")/UDP(sport=53, dport=61234),
        IP(src="172.16.0.4", dst="192.168.1.10")/TCP(sport=80, dport=45322),
    ]
    
    # Enregistrement du fichier
    wrpcap("data/sample.pcap", paquets)
    print("✅ Le fichier 'data/sample.pcap' a été créé localement avec succès ! (Pas besoin d'Internet)")

if __name__ == "__main__":
    creer_faux_pcap()
