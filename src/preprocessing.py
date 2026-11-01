# OWNER: Models team, coordinated with the PCAP-to-rows team.
# WHAT THIS FILE CONTAINS: Input-schema checks and the inference-time transform
# using the exact saved objects created by the model workflow. This includes
# the agreed cleaning/fill behavior, network_bytes = sbytes + dbytes,
# log1p-transformed numeric fields, the training-fitted scaler, and fitted
# proto/service/state encoders. Return features in the estimator's saved order
# (CSR/sparse where appropriate).
# WHAT IT MUST NOT DO: Fit or refit transformations on uploaded, validation, or
# held-out test data; silently fabricate missing fields; or duplicate logic in
# app.py. Coordinate any raw-field meaning changes with the PCAP team.
