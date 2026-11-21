from scapy.all import IP, IPv6, TCP, UDP, PcapReader


def lire_pcap(chemin_fichier):
    """Extract packet-level observations from a PCAP without guessing flow fields."""
    print(f"Lecture du fichier PCAP : {chemin_fichier}...")
    packets = []

    with PcapReader(str(chemin_fichier)) as pcap_reader:
        for packet in pcap_reader:
            if IP in packet:
                ip = packet[IP]
                ip_src = ip.src
                ip_dst = ip.dst
                ttl = int(ip.ttl)
                protocol_number = int(ip.proto)
            elif IPv6 in packet:
                ip = packet[IPv6]
                ip_src = ip.src
                ip_dst = ip.dst
                ttl = int(ip.hlim)
                protocol_number = int(ip.nh)
            else:
                continue

            port_src = None
            port_dst = None
            tcp_window = None
            tcp_sequence = None
            if TCP in packet:
                transport = packet[TCP]
                port_src = int(transport.sport)
                port_dst = int(transport.dport)
                tcp_window = int(transport.window)
                tcp_sequence = int(transport.seq)
                protocol = "tcp"
            elif UDP in packet:
                transport = packet[UDP]
                port_src = int(transport.sport)
                port_dst = int(transport.dport)
                protocol = "udp"
            else:
                protocol = str(protocol_number)

            packets.append({
                "srcip": ip_src,
                "sport": port_src,
                "dstip": ip_dst,
                "dsport": port_dst,
                "proto": protocol,
                "packet_bytes": len(packet),
                "timestamp": float(packet.time),
                "ttl": ttl,
                "tcp_window": tcp_window,
                "tcp_sequence": tcp_sequence,
            })

    print(f"{len(packets)} paquets IP extraits.")
    return packets
