MODEL_READINESS_REASON = (
    "PCAP extraction does not reproduce all required UNSW-NB15 features "
    "with the training definitions; do not pass this row to the saved model."
)


def _new_flow(packet):
    return {
        "srcip": packet["srcip"],
        "sport": packet["sport"],
        "dstip": packet["dstip"],
        "dsport": packet["dsport"],
        "proto": packet["proto"],
        "start_time": packet["timestamp"],
        "end_time": packet["timestamp"],
        "spkts": 0,
        "dpkts": 0,
        "sbytes": 0,
        "dbytes": 0,
        "sttl": None,
        "dttl": None,
        "swin": None,
        "dwin": None,
        "stcpb": None,
        "dtcpb": None,
        "_s_packet_size_total": 0,
        "_d_packet_size_total": 0,
    }


def _add_packet(flow, packet, direction):
    packet_size = packet["packet_bytes"]
    count_key = "spkts" if direction == "forward" else "dpkts"
    bytes_key = "sbytes" if direction == "forward" else "dbytes"
    ttl_key = "sttl" if direction == "forward" else "dttl"
    window_key = "swin" if direction == "forward" else "dwin"
    sequence_key = "stcpb" if direction == "forward" else "dtcpb"
    size_total_key = (
        "_s_packet_size_total" if direction == "forward" else "_d_packet_size_total"
    )

    flow[count_key] += 1
    flow[bytes_key] += packet_size
    flow[size_total_key] += packet_size
    if flow[ttl_key] is None:
        flow[ttl_key] = packet["ttl"]
    if flow[window_key] is None and packet["tcp_window"] is not None:
        flow[window_key] = packet["tcp_window"]
    if flow[sequence_key] is None and packet["tcp_sequence"] is not None:
        flow[sequence_key] = packet["tcp_sequence"]
    flow["start_time"] = min(flow["start_time"], packet["timestamp"])
    flow["end_time"] = max(flow["end_time"], packet["timestamp"])


def grouper_en_flux(liste_paquets):
    """Aggregate observed bidirectional packet facts; leave unknown features absent."""
    print("Agrégation des paquets en flux bidirectionnels...")
    flux_dict = {}

    for packet in liste_paquets:
        key_forward = (
            packet["srcip"], packet["sport"], packet["dstip"],
            packet["dsport"], packet["proto"],
        )
        key_reverse = (
            packet["dstip"], packet["dsport"], packet["srcip"],
            packet["sport"], packet["proto"],
        )

        if key_forward in flux_dict:
            flow = flux_dict[key_forward]
            direction = "forward"
        elif key_reverse in flux_dict:
            flow = flux_dict[key_reverse]
            direction = "reverse"
        else:
            flow = _new_flow(packet)
            flux_dict[key_forward] = flow
            direction = "forward"

        _add_packet(flow, packet, direction)

    flows = []
    for flow in flux_dict.values():
        forward_packets = flow["spkts"]
        reverse_packets = flow["dpkts"]
        flow["dur"] = max(0.0, flow["end_time"] - flow["start_time"])
        flow["stime"] = flow["start_time"]
        flow["ltime"] = flow["end_time"]
        flow["smeansz"] = (
            flow["_s_packet_size_total"] / forward_packets if forward_packets else None
        )
        flow["dmeansz"] = (
            flow["_d_packet_size_total"] / reverse_packets if reverse_packets else None
        )
        flow["network_bytes"] = flow["sbytes"] + flow["dbytes"]
        flow["model_ready"] = False
        flow["model_readiness_reason"] = MODEL_READINESS_REASON
        del flow["_s_packet_size_total"]
        del flow["_d_packet_size_total"]
        del flow["start_time"]
        del flow["end_time"]
        flows.append(flow)

    print(f"{len(flows)} flux bidirectionnels agrégés.")
    return flows
