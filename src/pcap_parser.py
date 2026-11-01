# OWNER: PCAP-to-rows team.
# WHAT THIS FILE CONTAINS: Functions that accept an uploaded PCAP/PCAPNG,
# decode packets offline, group packets into bidirectional flows, and return
# one row per flow in a pandas DataFrame (or the interface agreed with the app).
# Document flow keys, source/destination direction, idle/active timeouts,
# packet/byte counting, units, and errors for malformed/unsupported captures.
# WHAT IT MUST VALIDATE: Each returned model-input field must have documented
# semantics compatible with the model's training data.
# WHAT IT MUST NOT DO: Render UI, load/train a classifier, or invent missing
# ct_* counts, service, state, or other values to force a prediction.
