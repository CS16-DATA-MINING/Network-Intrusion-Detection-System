# OWNER: Models team.
# WHAT THIS FILE CONTAINS: Helpers to locate/load the trusted, fitted estimator
# artifact and a small prediction interface for validated flow records. The
# interface should use preprocessing.py, preserve the saved 55-feature order
# for the current feature-selected forest, and return class labels plus an
# optional supported decision/probability score.
# WHAT IT MUST NOT DO: Train/tune/retrain in response to an app request or load
# model files supplied by users. Keep estimator, preprocessing, selected
# features, and version metadata from the same artifact release.
