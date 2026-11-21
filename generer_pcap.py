from pathlib import Path

from scapy.all import IP, TCP, UDP, Raw, wrpcap


def creer_faux_pcap(chemin_sortie="data/sample.pcap"):
    output_path = Path(chemin_sortie)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = 1_700_000_000.0
    packets = [
        IP(src="192.168.1.10", dst="10.0.0.5", ttl=64)
        / TCP(sport=443, dport=50210, seq=1000, window=8192, flags="PA")
        / Raw(load=b"request"),
        IP(src="10.0.0.5", dst="192.168.1.10", ttl=61)
        / TCP(sport=50210, dport=443, seq=2000, window=4096, flags="PA")
        / Raw(load=b"response"),
        IP(src="192.168.1.10", dst="10.0.0.5", ttl=64)
        / TCP(sport=443, dport=50210, seq=1007, window=8192, flags="A"),
        IP(src="192.168.1.22", dst="8.8.8.8", ttl=64)
        / UDP(sport=53000, dport=53)
        / Raw(load=b"query"),
    ]
    for offset, packet in enumerate(packets):
        packet.time = start_time + offset * 0.25

    wrpcap(str(output_path), packets)
    print(f"PCAP de test local créé : {output_path}")
    return output_path


if __name__ == "__main__":
    creer_faux_pcap()
