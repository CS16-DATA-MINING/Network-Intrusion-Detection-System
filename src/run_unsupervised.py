"""Étape non supervisée : clustering, détection d'anomalies et règles d'association.

Usage : ``python -m src.run_unsupervised``
Sorties : results/tables/clustering_*.csv, anomaly_*.csv, association_rules*.csv,
          results/unsupervised.json (données pour l'application),
          models/unsupervised.joblib (préprocesseur + PCA + K-Means + détecteurs).
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from src import config as C
from src.preprocessing import build_preprocessor, feature_frame
from src.unsupervised import association as AR
from src.unsupervised import clustering as CL

warnings.filterwarnings("ignore")
RS = C.RANDOM_STATE


def _best_f1_threshold(y, score):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.0, float(f1[i]), float(p[i]), float(r[i])


def main():
    t0 = time.time()
    C.ensure_dirs()
    train = pd.read_csv(C.TRAIN_FILE, low_memory=False)
    test = pd.read_csv(C.TEST_FILE, low_memory=False)
    out: dict = {}

    # ------------------------------------------------------------------ données
    sub = train.sample(20000, random_state=RS).reset_index(drop=True)
    prep = build_preprocessor().fit(feature_frame(train))
    Xs = prep.transform(feature_frame(sub))
    pca = PCA(n_components=10, random_state=RS).fit(Xs)
    Xp = pca.transform(Xs)
    out["pca_explained"] = pca.explained_variance_ratio_.round(4).tolist()
    # l'espace de clustering = 10 composantes principales (≈ variance expliquée cumulée)
    Xc = Xp
    y_bin, y_cat = sub["label"].to_numpy(), sub["attack_cat"].to_numpy()
    print(f"PCA : {pca.explained_variance_ratio_.sum():.1%} de variance sur 10 composantes")

    # ---------------------------------------------------------------- K-Means
    ks = range(2, 13)
    scan_pp = CL.kmeans_scan(Xc, ks, "k-means++")
    scan_rand = CL.kmeans_scan(Xc, ks, "random")
    km_scan = pd.concat([scan_pp, scan_rand], ignore_index=True)
    km_scan.to_csv(C.METRICS_DIR / "clustering_kmeans_scan.csv", index=False)
    k_elbow = CL.elbow_k(scan_pp["inertia_wcss"].to_numpy(), ks)
    k_sil = int(scan_pp.loc[scan_pp["silhouette"].idxmax(), "k"])
    k_best = k_elbow
    print(f"K-Means : coude k={k_elbow}, silhouette max k={k_sil}")
    km = KMeans(n_clusters=k_best, n_init=10, random_state=RS).fit(Xc)
    km10 = KMeans(n_clusters=10, n_init=10, random_state=RS).fit(Xc)

    comp = []

    def record(name, labels, X=Xc, extra=None):
        r = {"algorithm": name, "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
             "noise_pct": float((np.asarray(labels) == -1).mean() * 100)}
        r.update(CL.internal_metrics(X, np.asarray(labels)))
        yb = y_bin[: len(labels)] if X is Xc else extra["y_bin"]
        yc = y_cat[: len(labels)] if X is Xc else extra["y_cat"]
        e = CL.external_metrics(yc, labels)
        r.update({f"{k}_cat": v for k, v in e.items()})
        r["purity_binary"] = CL.purity(yb, labels)
        comp.append(r)

    record(f"K-Means++ (k={k_best}, coude)", km.labels_)
    record("K-Means++ (k=10 = nb catégories)", km10.labels_)

    # ------------------------------------------------- PAM / CLARA (K-Medoids)
    n_small = 1500
    idx_s = np.random.default_rng(RS).choice(len(Xc), n_small, replace=False)
    Xsm = Xc[idx_s]
    ext_small = {"y_bin": y_bin[idx_s], "y_cat": y_cat[idx_s]}
    med, lab_pam, cost = CL.pam(Xsm, k_best)
    record(f"K-Medoids PAM (k={k_best}, n={n_small})", lab_pam, Xsm, ext_small)
    km_small = KMeans(n_clusters=k_best, n_init=10, random_state=RS).fit(Xsm)
    record(f"K-Means++ (k={k_best}, n={n_small}, même échantillon)", km_small.labels_, Xsm, ext_small)
    centers_clara, lab_clara, cost_clara = CL.clara(Xc, k_best, n_samples=5, sample_size=800)
    record(f"CLARA (k={k_best}, 5×800)", lab_clara)
    # robustesse aux valeurs aberrantes : déplacement des centres si on injecte 2 % d'outliers
    rng = np.random.default_rng(RS)
    n_out = int(0.02 * n_small)
    X_out = np.vstack([Xsm, Xsm[rng.choice(n_small, n_out)] * 15])
    med_o, _, _ = CL.pam(X_out, k_best)
    km_o = KMeans(n_clusters=k_best, n_init=10, random_state=RS).fit(X_out)

    def shift(a, b):
        from scipy.optimize import linear_sum_assignment
        d = np.linalg.norm(a[:, None] - b[None], axis=2)
        r, c = linear_sum_assignment(d)
        return float(d[r, c].mean())
    out["robustness"] = {
        "pam_center_shift": shift(Xsm[med], X_out[med_o]),
        "kmeans_center_shift": shift(km_small.cluster_centers_, km_o.cluster_centers_),
        "outliers_injected": n_out}
    print("Robustesse :", out["robustness"])

    # ------------------------------------------------------- AGNES et DIANA
    n_h = 2000
    idx_h = np.random.default_rng(RS + 1).choice(len(Xc), n_h, replace=False)
    Xh = Xc[idx_h]
    ext_h = {"y_bin": y_bin[idx_h], "y_cat": y_cat[idx_h]}
    agnes_tab, links = CL.agnes(Xh, k=k_best)
    agnes_tab.to_csv(C.METRICS_DIR / "clustering_agnes.csv", index=False)
    for m, Z in links.items():
        record(f"AGNES {m} (k={k_best}, n={n_h})", fcluster(Z, k_best, "maxclust") - 1, Xh, ext_h)
    joblib.dump({"ward": links["ward"], "complete": links["complete"],
                 "average": links["average"], "single": links["single"],
                 "y_cat": ext_h["y_cat"]}, C.ALL_MODELS_DIR / "linkages.joblib", compress=3)
    n_d = 700
    Xd = Xh[:n_d]
    lab_diana, hist = CL.diana(Xd, k_best)
    record(f"DIANA (k={k_best}, n={n_d})", lab_diana, Xd,
           {"y_bin": ext_h["y_bin"][:n_d], "y_cat": ext_h["y_cat"][:n_d]})
    out["diana_history"] = hist

    # ---------------------------------------------------------------- DBSCAN
    kd = CL.k_distance(Xc[:8000], k=10)
    out["k_distance"] = kd[:: max(1, len(kd) // 300)].round(4).tolist()
    eps_knee = float(np.quantile(kd, 0.90))
    eps_vals = np.round(np.unique(np.r_[np.quantile(kd, [0.6, 0.75, 0.9, 0.95]), eps_knee]), 3)
    db_scan = CL.dbscan_scan(Xc[:8000], eps_vals, (5, 10, 20))
    db_scan.to_csv(C.METRICS_DIR / "clustering_dbscan_scan.csv", index=False)
    valid = db_scan[(db_scan.n_clusters >= 2) & (db_scan.noise_pct < 20)]
    best_db = (valid if len(valid) else db_scan).sort_values("silhouette", ascending=False).iloc[0]
    lab_db = DBSCAN(eps=best_db.eps, min_samples=int(best_db.min_samples)).fit_predict(Xc[:8000])
    record(f"DBSCAN (eps={best_db.eps:.2f}, minPts={int(best_db.min_samples)})", lab_db, Xc[:8000],
           {"y_bin": y_bin[:8000], "y_cat": y_cat[:8000]})

    # ------------------------------------------------------------------- GMM
    gmm_tab = CL.gmm_scan(Xc[:10000], ks)
    gmm_tab.to_csv(C.METRICS_DIR / "clustering_gmm_scan.csv", index=False)
    k_gmm = int(gmm_tab.loc[gmm_tab["bic"].idxmin(), "k"])
    gmm = GaussianMixture(n_components=k_gmm, random_state=RS, reg_covar=1e-4).fit(Xc[:10000])
    record(f"GMM (k={k_gmm}, BIC min)", gmm.predict(Xc[:10000]), Xc[:10000],
           {"y_bin": y_bin[:10000], "y_cat": y_cat[:10000]})

    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(C.METRICS_DIR / "clustering_comparison.csv", index=False)
    print(comp_df[["algorithm", "n_clusters", "silhouette", "davies_bouldin", "adjusted_rand_cat",
                   "purity_cat", "purity_binary"]].round(3).to_string())

    # profils des clusters K-Means (interprétation)
    prof = sub.assign(cluster=km.labels_)
    ct = pd.crosstab(prof["cluster"], prof["attack_cat"])
    ct.to_csv(C.METRICS_DIR / "clustering_kmeans_vs_category.csv")
    prof_tab = prof.groupby("cluster").agg(
        taille=("label", "size"), pct_attaque=("label", "mean"),
        sbytes_med=("sbytes", "median"), dbytes_med=("dbytes", "median"),
        dur_med=("dur", "median"), sttl_med=("sttl", "median"),
        proto_mode=("proto", lambda s: s.mode().iat[0]),
        service_mode=("service", lambda s: s.mode().iat[0]),
        state_mode=("state", lambda s: s.mode().iat[0]),
        categorie_dominante=("attack_cat", lambda s: s.mode().iat[0]))
    prof_tab.to_csv(C.METRICS_DIR / "clustering_kmeans_profiles.csv")

    # --------------------------------------------- détection d'anomalies
    Xtr_all = prep.transform(feature_frame(train))
    Xte = prep.transform(feature_frame(test))
    yte = test["label"].to_numpy()
    Ptr, Pte = pca.transform(Xtr_all), pca.transform(Xte)
    normal_tr = Ptr[train["label"].to_numpy() == 0]
    anom = []
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=RS, n_jobs=-1).fit(Ptr)
    s_iso = -iso.score_samples(Pte)
    iso_n = IsolationForest(n_estimators=300, random_state=RS, n_jobs=-1).fit(normal_tr)
    s_iso_n = -iso_n.score_samples(Pte)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(normal_tr[:20000])
    s_lof = -lof.score_samples(Pte)
    km_n = KMeans(n_clusters=10, n_init=5, random_state=RS).fit(normal_tr)
    s_km = km_n.transform(Pte).min(axis=1)
    gmm_n = GaussianMixture(n_components=8, random_state=RS, reg_covar=1e-4).fit(normal_tr[:20000])
    s_gmm = -gmm_n.score_samples(Pte)
    for name, mode, s in [("Isolation Forest", "non supervisé (train sans étiquettes)", s_iso),
                          ("Isolation Forest", "semi-supervisé (normal seulement)", s_iso_n),
                          ("Local Outlier Factor", "semi-supervisé (normal seulement)", s_lof),
                          ("Distance au centroïde K-Means", "semi-supervisé (normal seulement)", s_km),
                          ("Log-vraisemblance GMM", "semi-supervisé (normal seulement)", s_gmm)]:
        thr, f1, p, r = _best_f1_threshold(yte, s)
        anom.append({"detector": name, "mode": mode, "roc_auc": float(roc_auc_score(yte, s)),
                     "pr_auc": float(average_precision_score(yte, s)), "best_f1": f1,
                     "precision_at_best": p, "recall_at_best": r, "threshold": thr})
    anom_df = pd.DataFrame(anom)
    anom_df.to_csv(C.METRICS_DIR / "anomaly_detection.csv", index=False)
    print(anom_df.round(4).to_string())
    iso_thr = float(anom_df.iloc[1]["threshold"])

    # ---------------------------------------------- règles d'association
    tr_rules = train.sample(15000, random_state=RS)
    trans = AR.discretize(tr_rules)
    freq = AR.apriori(trans, min_support=0.03, max_len=4)
    rules_all = AR.association_rules(freq, min_confidence=0.7)
    rules_cls = AR.association_rules(freq, min_confidence=0.8, consequent_prefix="classe=")
    rules_all.head(300).to_csv(C.METRICS_DIR / "association_rules_all.csv", index=False)
    rules_cls.to_csv(C.METRICS_DIR / "association_rules_class.csv", index=False)
    trans_c = AR.discretize(train[train.label == 1].sample(12000, random_state=RS), target="attack_cat")
    freq_c = AR.apriori(trans_c, min_support=0.02, max_len=4)
    rules_cat = AR.association_rules(freq_c, min_confidence=0.6, consequent_prefix="classe=")
    rules_cat.to_csv(C.METRICS_DIR / "association_rules_category.csv", index=False)
    print(f"Apriori : {len(freq)} itemsets fréquents, {len(rules_all)} règles (conf≥0.7), "
          f"{len(rules_cls)} règles vers la classe, {len(rules_cat)} vers une catégorie")

    # ------------------------------------------------- export pour l'application
    rng = np.random.default_rng(RS)
    pts = rng.choice(len(Xp), 3000, replace=False)
    out.update({
        "k_elbow": k_elbow, "k_silhouette": k_sil, "k_best": k_best, "k_gmm": k_gmm,
        "kmeans_scan": km_scan.to_dict(orient="records"),
        "agnes": agnes_tab.to_dict(orient="records"),
        "dbscan_scan": db_scan.to_dict(orient="records"),
        "gmm_scan": gmm_tab.to_dict(orient="records"),
        "comparison": comp_df.to_dict(orient="records"),
        "cluster_vs_category": {"index": ct.index.tolist(), "columns": ct.columns.tolist(),
                                "values": ct.to_numpy().tolist()},
        "profiles": prof_tab.reset_index().to_dict(orient="records"),
        "anomaly": anom_df.to_dict(orient="records"),
        "pca_points": {"x": Xp[pts, 0].round(3).tolist(), "y": Xp[pts, 1].round(3).tolist(),
                       "cluster": km.labels_[pts].tolist(), "attack_cat": y_cat[pts].tolist(),
                       "label": y_bin[pts].tolist()},
        "rules_class": rules_cls.head(60).replace([np.inf], 999).to_dict(orient="records"),
        "rules_category": rules_cat.head(60).replace([np.inf], 999).to_dict(orient="records"),
        "n_frequent_itemsets": len(freq), "n_rules": int(len(rules_all)),
        "elapsed_s": round(time.time() - t0, 1),
    })
    (C.METRICS_DIR / "unsupervised.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    joblib.dump({"prep": prep, "pca": pca, "kmeans": km, "iforest": iso_n,
                 "iforest_threshold": iso_thr, "kmeans_normal": km_n,
                 "cluster_profiles": prof_tab.reset_index().to_dict(orient="records")},
                C.ALL_MODELS_DIR / "unsupervised.joblib", compress=3)
    print(f"Terminé en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
