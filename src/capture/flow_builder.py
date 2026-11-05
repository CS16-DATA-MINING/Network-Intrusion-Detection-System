def grouper_en_flux(liste_paquets):
    print("Agrégation des paquets en flux bidirectionnels...")
    flux_dict = {}

    for pkt in liste_paquets:

        cle_aller = (pkt['ip_src'], pkt['port_src'], pkt['ip_dst'], pkt['port_dst'], pkt['proto'])
        cle_retour = (pkt['ip_dst'], pkt['port_dst'], pkt['ip_src'], pkt['port_src'], pkt['proto'])

        if cle_aller in flux_dict:
            flux = flux_dict[cle_aller]
            flux['spkts'] += 1
            flux['sbytes'] += pkt['taille']
            flux['dur'] = pkt['timestamp'] - flux['start_time']
        elif cle_retour in flux_dict:
            flux = flux_dict[cle_retour]
            flux['dpkts'] += 1
            flux['dbytes'] += pkt['taille']
            flux['dur'] = pkt['timestamp'] - flux['start_time']
        else:
            flux_dict[cle_aller] = {
                'dur': 0.0, 'proto': pkt['proto'], 'spkts': 1, 'dpkts': 0,
                'sbytes': pkt['taille'], 'dbytes': 0, 'start_time': pkt['timestamp']
            }

    liste_flux_finaux = list(flux_dict.values())
    print(f"➔ {len(liste_flux_finaux)} flux réseau bidirectionnels identifiés.")
    return liste_flux_finaux
