# OWNER: App development team.
# WHAT THIS FILE CONTAINS: The Streamlit page layout and user interaction:
# PCAP/PCAPNG upload, analyze action, flow-result table/summary, clear messages
# for invalid or unsupported input, and a separate labeled-CSV evaluation page.
# It calls pcap_parser.py to obtain flow rows, then models.py (through
# preprocessing.py as agreed) to obtain predictions, and evaluation.py for
# metrics when ground-truth labels are supplied.
# WHAT IT DOES NOT CONTAIN: Packet decoding, feature engineering, model fitting,
# or duplicated preprocessing. While working independently, use only clearly
# labeled synthetic mocks; replace them with the real team interfaces before
# presenting predictions as model results.
