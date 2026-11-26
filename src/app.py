"""Application web NIDS — point d'entrée Flask.

Lancement : ``python -m src.app`` (ou ``python app.py`` à la racine)
puis ouvrir http://127.0.0.1:5000

Pages : accueil, dashboard EDA, tests statistiques, modèles, explicabilité,
prédiction temps réel, analyse PCAP, non supervisé, règles d'association,
génération des tuples (extracteur), documentation. Toutes les données des
graphiques sont servies en JSON par l'API ``/api/*`` (voir ``src/services.py``).
"""
from __future__ import annotations

import io
import tempfile
import traceback
from pathlib import Path

from flask import (Flask, abort, jsonify, render_template, request, send_file,
                   send_from_directory)
from werkzeug.utils import secure_filename

from src import config as C
from src import services as S

ALLOWED = {".pcap", ".pcapng", ".cap", ".csv"}
MAX_UPLOAD_MB = 200

PAGES = [
    ("index", "/", "Accueil", "home"),
    ("eda", "/eda", "Dashboard EDA", "chart"),
    ("stats", "/stats", "Tests statistiques", "sigma"),
    ("models", "/models", "Modèles & résultats", "trophy"),
    ("importance", "/importance", "Explicabilité", "bulb"),
    ("predict", "/predict", "Prédiction temps réel", "bolt"),
    ("pcap", "/pcap", "Analyse PCAP", "radar"),
    ("clustering", "/clustering", "Non supervisé", "cluster"),
    ("rules", "/rules", "Règles d'association", "link"),
    ("extractor", "/extractor", "Génération des tuples", "cpu"),
    ("about", "/about", "Documentation", "book"),
]


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.json.ensure_ascii = False
    app.json.sort_keys = False          # conserve l'ordre logique (catégories, modèles)

    @app.context_processor
    def inject_nav():
        return {"pages": PAGES, "current": request.endpoint}

    def page(endpoint, template, **ctx):
        def view():
            return render_template(template, **ctx)
        view.__name__ = endpoint
        return view

    for ep, url, _, _ in PAGES:
        app.add_url_rule(url, ep, page(ep, f"{ep}.html"))

    # -------------------------------------------------------------- statiques
    @app.route("/vendor/plotly.min.js")
    def plotly_js():
        """Plotly.js servi depuis le paquet Python installé (fonctionne hors-ligne)."""
        import plotly
        d = Path(plotly.__file__).parent / "package_data"
        return send_from_directory(d, "plotly.min.js", max_age=86400)

    @app.route("/figures/<path:name>")
    def figures(name):
        return send_from_directory(C.FIGURES_DIR, name)   # ex. /figures/eda/01_missing_values.png

    # -------------------------------------------------------------- erreurs
    @app.errorhandler(413)
    def too_big(_):
        return jsonify(error=f"Fichier trop volumineux (max {MAX_UPLOAD_MB} Mo)."), 413

    def fail(exc: Exception, code: int = 400):
        app.logger.warning("Erreur API : %s\n%s", exc, traceback.format_exc())
        return jsonify(error=str(exc)), code

    # -------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        reg = S.registry()
        return jsonify(status="ok" if reg else "models_missing",
                       models=bool(reg), dataset=C.CLEAN_FILE.exists(),
                       unsupervised=(C.ALL_MODELS_DIR / "unsupervised.joblib").exists(),
                       analysis=(C.METRICS_DIR / "analysis.json").exists())

    @app.get("/api/overview")
    def api_overview():
        return jsonify(S.overview())

    @app.get("/api/eda/options")
    def api_eda_options():
        return jsonify(S.eda_options())

    @app.post("/api/eda")
    def api_eda():
        try:
            return jsonify(S.eda_payload(request.get_json(force=True) or {}))
        except Exception as e:  # noqa: BLE001
            return fail(e)

    @app.get("/api/eda/missing")
    def api_missing():
        return jsonify(S.missing_payload())

    @app.get("/api/stats")
    def api_stats():
        names = {"normality": "stat_normality.csv", "mann_whitney": "stat_mann_whitney.csv",
                 "kruskal": "stat_kruskal_wallis.csv", "chi2": "stat_chi2.csv",
                 "spearman": "stat_spearman_target.csv", "outliers": "eda_outliers_iqr.csv"}
        out = {k: S.records(S.table(v)) for k, v in names.items()}
        sm = S.table("stat_spearman_matrix.csv")
        if len(sm):
            sm = sm.set_index(sm.columns[0])
            out["spearman_matrix"] = {"vars": sm.columns.tolist(),
                                      "matrix": S.clean_json(sm.round(3).to_numpy().tolist())}
        out["summary"] = S.analysis().get("eda", {})
        return jsonify(S.clean_json(out))

    @app.get("/api/models")
    def api_models():
        task = request.args.get("task", "binary")
        profile = request.args.get("profile", "full")
        if task not in ("binary", "multiclass") or profile not in ("full", "portable"):
            abort(400)
        return jsonify(S.models_payload(task, profile))

    @app.get("/api/models/list")
    def api_models_list():
        return jsonify(S.available_models(request.args.get("task", "binary"),
                                          request.args.get("profile", "full")))

    @app.get("/api/model-info")
    def api_model_info():
        reg = S.registry()
        return jsonify(S.clean_json({k: reg.get(k) for k in ("best", "class_names", "features",
                                                            "ttl_features", "versions", "created",
                                                            "train_rows", "test_rows")}))

    @app.get("/api/importance")
    def api_importance():
        an = S.analysis()
        sh = S.table("shap_multiclass_by_category.csv")
        out = {"importance": S.records(S.table("feature_importance.csv")),
               "agreement": S.records(S.table("feature_importance_rank_agreement.csv")),
               "ablation": an.get("ablation", []), "descriptions": C.FEATURE_DESCRIPTIONS}
        if len(sh):
            sh = sh.set_index(sh.columns[0])
            top = sh.sum(axis=1).sort_values(ascending=False).index[:18]
            sh = sh.loc[top]
            out["shap_by_category"] = {"features": top.tolist(), "classes": sh.columns.tolist(),
                                       "values": S.clean_json(sh.round(4).to_numpy().tolist())}
        return jsonify(S.clean_json(out))

    @app.get("/api/presets")
    def api_presets():
        return jsonify(S.presets())

    @app.post("/api/predict")
    def api_predict():
        try:
            body = request.get_json(force=True) or {}
            vals = body.get("features", body)
            return jsonify(S.predict_one(vals, body.get("model") or None,
                                         body.get("profile", "full"),
                                         float(body.get("threshold", 0.5))))
        except Exception as e:  # noqa: BLE001
            return fail(e)

    @app.post("/api/pcap")
    def api_pcap():
        try:
            profile = request.form.get("profile", "full")
            model = request.form.get("model") or None
            threshold = float(request.form.get("threshold", 0.5))
            ct_mode = request.form.get("ct_mode", "block")
            demo = request.form.get("demo")
            if demo:
                name = secure_filename(demo)
                path = C.SAMPLE_PCAP_DIR / f"{name}.pcap"
                if not path.exists():
                    raise FileNotFoundError("PCAP de démonstration introuvable : lancez "
                                            "`python -m src.pcap.generator`.")
                labels = C.SAMPLE_PCAP_DIR / f"{name}_labels.csv"
                return jsonify(S.analyze_capture(path, path.name, profile, model, threshold,
                                                 ct_mode, labels))
            f = request.files.get("file")
            if f is None or not f.filename:
                raise ValueError("Aucun fichier reçu.")
            suffix = Path(f.filename).suffix.lower()
            if suffix not in ALLOWED:
                raise ValueError(f"Format non supporté ({suffix}). Formats acceptés : "
                                 + ", ".join(sorted(ALLOWED)))
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / ("capture" + suffix)
                f.save(p)
                if p.stat().st_size == 0:
                    raise ValueError("Le fichier est vide.")
                return jsonify(S.analyze_capture(p, secure_filename(f.filename), profile, model,
                                                 threshold, ct_mode))
        except Exception as e:  # noqa: BLE001
            return fail(e)

    @app.get("/api/pcap/<job>/flow/<int:flow_id>")
    def api_pcap_flow(job, flow_id):
        try:
            return jsonify(S.explain_flow(job, flow_id))
        except KeyError as e:
            return fail(e, 404)

    @app.get("/api/pcap/<job>/download")
    def api_pcap_download(job):
        try:
            df = S.job_frame(job)
        except KeyError as e:
            return fail(e, 404)
        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        return send_file(buf, mimetype="text/csv", as_attachment=True,
                         download_name=f"flux_unsw47_predictions_{job}.csv")

    @app.get("/api/unsupervised")
    def api_unsup():
        return jsonify(S.clean_json(S.unsupervised()))

    @app.get("/api/dendrogram")
    def api_dendrogram():
        return jsonify(S.dendrogram_payload(request.args.get("method", "ward")))

    @app.get("/api/rules")
    def api_rules():
        return jsonify({"class": S.records(S.table("association_rules_class.csv")),
                        "category": S.records(S.table("association_rules_category.csv")),
                        "all": S.records(S.table("association_rules_all.csv").head(300))})

    @app.get("/api/extractor")
    def api_extractor():
        an = S.analysis()
        groups = []
        sources = {"flow": "Identifiants de flux (Argus + Bro)", "basic": "Argus (features de base)",
                   "content": "Argus + Bro (features de contenu)", "time": "Argus (features temporelles)",
                   "additional_general": "Features additionnelles générales (Alg. 3.2–3.6)",
                   "additional_connection": "Features de connexion, fenêtres de 100 (Alg. 3.7–3.9)"}
        for g, cols in C.FEATURE_GROUPS.items():
            groups.append({"group": g, "source": sources[g],
                           "features": [{"name": c, "description": C.FEATURE_DESCRIPTIONS.get(c, ""),
                                         "used": c in C.FEATURES} for c in cols]})
        return jsonify(S.clean_json({"groups": groups, "fidelity": an.get("fidelity", []),
                                     "pcap_e2e": an.get("pcap_e2e", []),
                                     "replay_fidelity": an.get("replay_fidelity", []),
                                     "demos": sorted(p.stem for p in C.SAMPLE_PCAP_DIR.glob("*.pcap"))}))

    @app.get("/api/sample-csv")
    def api_sample_csv():
        """Petit CSV de flux du jeu de test (avec vérité terrain) pour tester l'import CSV."""
        import pandas as pd
        df = pd.read_csv(C.TEST_FILE, low_memory=False).sample(300, random_state=1)
        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        return send_file(buf, mimetype="text/csv", as_attachment=True,
                         download_name="exemple_flux_unsw_nb15.csv")

    return app


app = create_app()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, debug=a.debug)
