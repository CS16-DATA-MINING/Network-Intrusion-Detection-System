# OWNER: Models team (metric definitions); App team (evaluation display).
# WHAT THIS FILE CONTAINS: Reusable metric calculations from true labels,
# predicted classes, and optional continuous model scores. Include F1 and a
# confusion matrix from predicted classes; calculate ROC-AUC from probability
# or decision scores, not hard class predictions. Report unsupported metrics
# clearly when a class or score is unavailable.
# WHAT IT MUST NOT DO: Train/tune models or treat unlabeled PCAP predictions as
# evaluation. The held-out test labels are for final evaluation, never fitting.
