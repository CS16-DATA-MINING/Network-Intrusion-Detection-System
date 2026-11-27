"""Tests de l'application Flask : pages et API (ignorés si les artefacts sont absents)."""
import io

import pytest

from src import config as C

pytestmark = pytest.mark.skipif(not (C.REGISTRY_FILE).exists() or not C.TEST_FILE.exists(),
                                reason="lancer d'abord src.prepare_data et src.train")


@pytest.fixture(scope="module")
def client():
    from src.app import create_app
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.mark.parametrize("url", ["/", "/eda", "/stats", "/models", "/importance", "/predict", "/pcap",
                                 "/clustering", "/rules", "/extractor", "/about"])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200 and b"NIDS" in r.data


def test_api_core(client):
    assert client.get("/api/health").get_json()["models"] is True
    ov = client.get("/api/overview").get_json()
    assert ov["raw_total"] > 2_500_000 and ov["n_features"] == len(C.FEATURES)
    eda = client.post("/api/eda", json={"categories": ["DoS"], "num_var": "sbytes"}).get_json()
    assert eda["n_rows"] > 0 and eda["n_attack"] == eda["n_rows"]
    for k in ("describe", "hist", "cat_freq", "corr", "box"):
        assert k in eda
    assert client.get("/vendor/plotly.min.js").status_code == 200
    m = client.get("/api/models?task=binary&profile=full").get_json()
    assert len(m["rows"]) >= 10


def test_api_predict_preset(client):
    presets = client.get("/api/presets").get_json()
    name = next(n for n in presets if n.startswith("Generic"))
    r = client.post("/api/predict", json={"features": presets[name], "threshold": 0.5}).get_json()
    assert 0 <= r["proba"] <= 1 and r["prediction"] == 1
    assert r["explanation"]["contributions"] and r["text"]
    # l'utilisateur reçoit le type d'attaque + les probabilités des 10 catégories
    assert r["category"] in C.ATTACK_CATEGORIES[1:]
    assert set(r["multiclass"]) == set(C.ATTACK_CATEGORIES)
    assert abs(sum(r["multiclass"].values()) - 1) < 1e-3
    assert "Catégorie d'attaque" in r["text"]
    normal = client.post("/api/predict", json={"features": presets["Normal — DNS"]}).get_json()
    assert normal["prediction"] == 0 and normal["category"] == "Normal"


def test_api_pcap_demo_and_errors(client):
    r = client.post("/api/pcap", data={"demo": "demo_unsw_replay", "profile": "portable", "threshold": "0.5"})
    j = r.get_json()
    assert r.status_code == 200, j
    assert j["n_flows"] > 1000 and j["truth"]["f1"] > 0.85 and j["truth"]["recall"] > 0.95
    assert j["truth"]["unseen"]["n_attacks"] > 0 and j["truth"]["attack_type_accuracy"] > 0.6
    cats = {f["categorie_predite"] for f in j["flows"] if f["prediction"] == 1}
    assert cats and cats <= set(C.ATTACK_CATEGORIES[1:])          # type d'attaque pour chaque alerte
    assert "confusion_cat" in j["truth"] and sum(j["cat_counts"].values()) == j["n_flows"]
    assert client.get(f"/api/pcap/{j['job']}/download").status_code == 200
    assert "tuple" in client.get(f"/api/pcap/{j['job']}/flow/1").get_json()
    bad = client.post("/api/pcap", data={"file": (io.BytesIO(b"hello"), "x.txt")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400 and "Format" in bad.get_json()["error"]
    empty = client.post("/api/pcap", data={"file": (io.BytesIO(b"garbage-bytes"), "x.pcap")},
                        content_type="multipart/form-data")
    assert empty.status_code == 400
