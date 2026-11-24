"""Tests du prétraitement, des tests statistiques et des algorithmes non supervisés."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.evaluation import statistical_tests as ST
from src.preprocessing import (CorrelationFilter, SkewLog1p, build_preprocessor,
                               clean_dataframe, clean_training_data, feature_frame)
from src.unsupervised import association as AR
from src.unsupervised import clustering as CL


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.exponential(10, n) for c in C.NUMERIC_FEATURES})
    df["proto"] = rng.choice(["tcp", "udp"], n)
    df["state"] = rng.choice(["FIN", "CON", "INT"], n)
    df["service"] = rng.choice(["-", "http", "dns"], n)
    df["label"] = (df["sttl"] > 10).astype(int)
    df["attack_cat"] = np.where(df["label"] == 1, "Generic", "Normal")
    return df


def test_clean_dataframe_raw_quirks():
    raw = pd.DataFrame({"sport": ["0x000b", "-", "80"], "dsport": ["53", "0xc0a8", " "],
                        "ct_ftp_cmd": [" ", "2", None], "is_ftp_login": [None, 4, 1],
                        "ct_flw_http_mthd": [None, 1, 0], "service": ["-", "HTTP ", None],
                        "attack_cat": [None, " Fuzzers", "Backdoors"], "label": [0, 1, 1]})
    out = clean_dataframe(raw)
    assert out["sport"].tolist()[0] == 11 and np.isnan(out["sport"].tolist()[1])
    assert out["dsport"].tolist()[1] == 0xC0A8
    assert out["ct_ftp_cmd"].tolist() == [0, 2, 0]
    assert out["is_ftp_login"].tolist() == [0, 1, 1]
    assert out["service"].tolist() == ["-", "http", "-"]
    assert out["attack_cat"].tolist() == ["Normal", "Fuzzers", "Backdoor"]


def test_clean_training_removes_duplicates():
    df = _toy(50)
    dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
    out, rep = clean_training_data(dup)
    assert rep["duplicates_removed"] == 10 and len(out) == 50


def test_preprocessor_fit_on_train_handles_unknown():
    df = _toy()
    pre = build_preprocessor().fit(feature_frame(df))
    new = df.iloc[:3].copy()
    new.loc[:, "proto"] = "jamais-vu"
    Z = pre.transform(feature_frame(new))
    assert Z.shape[0] == 3 and np.isfinite(Z).all()
    # ttl retirés dans le profil portable
    pre2 = build_preprocessor(drop=C.TTL_FEATURES).fit(feature_frame(df))
    names = set(pre2.get_feature_names_out())
    assert not names & set(C.TTL_FEATURES)


def test_skew_and_correlation_filters():
    rng = np.random.default_rng(1)
    x = rng.exponential(1, (500, 1)) ** 3
    X = np.hstack([x, x * 2 + 1e-9, rng.normal(size=(500, 1))])
    f = CorrelationFilter(0.95).fit(X)
    assert f.keep_mask_.tolist() == [True, False, True]
    s = SkewLog1p(2.0).fit(X)
    assert s.log_mask_[0] and not s.log_mask_[2]


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = ST.benjamini_hochberg(p)
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_statistical_tests_detect_signal():
    df = _toy(600)
    mw = ST.mann_whitney(df, ["sttl", "dur"])
    assert mw.set_index("feature").loc["sttl", "p_bh"] < 1e-10
    chi = ST.chi2_tests(df, ["proto"])
    assert 0 <= chi.loc[0, "cramers_v"] <= 1
    mc = ST.mcnemar(np.array([1, 0, 1, 1]), np.array([1, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert mc["b"] == 1 and mc["c"] == 0


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [8, 8], [0, 8]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(40, 2)) for c in centers])
    return X, np.repeat([0, 1, 2], 40)


def test_pam_and_clara_recover_blobs():
    X, y = _blobs()
    med, lab, cost = CL.pam(X, 3)
    assert len(set(med)) == 3 and CL.external_metrics(y, lab)["adjusted_rand"] > 0.95
    _, lab2, _ = CL.clara(X, 3, n_samples=3, sample_size=60)
    assert CL.external_metrics(y, lab2)["adjusted_rand"] > 0.95


def test_diana_and_agnes():
    X, y = _blobs(1)
    lab, hist = CL.diana(X, 3)
    assert len(hist) == 2 and CL.external_metrics(y, lab)["adjusted_rand"] > 0.95
    tab, links = CL.agnes(X, k=3)
    assert set(tab["linkage"]) == {"ward", "complete", "average", "single"}
    assert (tab["cophenetic_corr"] > 0.5).all()


def test_elbow_detection():
    assert CL.elbow_k([1000, 400, 150, 130, 120, 115], range(2, 8)) == 4


def test_apriori_textbook_example():
    tx = [frozenset(t) for t in (["pain", "lait"], ["pain", "couches", "bière", "oeufs"],
                                 ["lait", "couches", "bière", "cola"], ["pain", "lait", "couches", "bière"],
                                 ["pain", "lait", "couches", "cola"])]
    freq = AR.apriori(tx, min_support=0.6)
    assert freq[frozenset(["pain", "lait"])] == pytest.approx(0.6)
    assert frozenset(["bière", "couches"]) in freq
    rules = AR.association_rules(freq, min_confidence=0.9)
    r = rules[(rules.antecedent == "bière") & (rules.consequent == "couches")].iloc[0]
    assert r.confidence == pytest.approx(1.0) and r.lift == pytest.approx(1.25)
