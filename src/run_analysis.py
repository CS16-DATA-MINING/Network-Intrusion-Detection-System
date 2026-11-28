"""Analyses complémentaires : EDA statistique, comparaison statistique des modèles,
importance des features (5 méthodes), courbe d'apprentissage, ablation par groupe
de features, fidélité de l'extracteur PCAP et évaluation bout-en-bout sur PCAP.

Usage : ``python -m src.run_analysis`` (après ``src.train``)
Sortie principale : results/analysis.json + tables CSV.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, learning_curve

from src import config as C
from src.evaluation import explain as EX
from src.evaluation import statistical_tests as ST
from src.models import load_registry
from src.preprocessing import clean_dataframe, feature_frame, read_raw_file

warnings.filterwarnings("ignore")
RS = C.RANDOM_STATE


def _load(task, key, profile="full"):
    reg = json.loads(C.REGISTRY_FILE.read_text(encoding="utf-8"))
    return joblib.load(C.ALL_MODELS_DIR / reg["models"][profile][task][key]["file"])


def eda_statistics(train: pd.DataFrame) -> dict:
    num = C.NUMERIC_FEATURES
    out = {}
    desc = train[num].describe().T
    desc["skew"] = train[num].skew()
    desc["kurtosis"] = train[num].kurt()
    desc["cv"] = desc["std"] / desc["mean"].replace(0, np.nan)
    desc.to_csv(C.METRICS_DIR / "eda_descriptive.csv")
    norm = ST.normality_tests(train, num)
    norm.to_csv(C.METRICS_DIR / "stat_normality.csv", index=False)
    mw = ST.mann_whitney(train, num)
    mw.to_csv(C.METRICS_DIR / "stat_mann_whitney.csv", index=False)
    kw = ST.kruskal_wallis(train, num)
    kw.to_csv(C.METRICS_DIR / "stat_kruskal_wallis.csv", index=False)
    chi = pd.concat([ST.chi2_tests(train, C.CATEGORICAL_FEATURES, "label"),
                     ST.chi2_tests(train, C.CATEGORICAL_FEATURES, "attack_cat")])
    chi.to_csv(C.METRICS_DIR / "stat_chi2.csv", index=False)
    sp = ST.spearman_with_target(train, num)
    sp.to_csv(C.METRICS_DIR / "stat_spearman_target.csv", index=False)
    ST.spearman_matrix(train, num).to_csv(C.METRICS_DIR / "stat_spearman_matrix.csv")
    out_iqr = ST.outlier_summary(train, num)
    out_iqr.to_csv(C.METRICS_DIR / "eda_outliers_iqr.csv", index=False)
    out.update({"n_normal_rejected": int((~norm["normal_at_5pct"]).sum()),
                "n_tests_normality": int(norm["shapiro_p"].notna().sum()),
                "mw_significant": int(mw["significant"].sum()), "mw_total": int(len(mw)),
                "kw_significant": int(kw["significant"].sum()), "kw_total": int(len(kw)),
                "top_mw": mw.head(10)[["feature", "rank_biserial_r", "p_bh"]].to_dict(orient="records"),
                "chi2": chi.to_dict(orient="records")})
    return out


def model_statistics(test: pd.DataFrame) -> dict:
    reg = json.loads(C.REGISTRY_FILE.read_text(encoding="utf-8"))
    res = {}
    for task in ("binary", "multiclass"):
        fs = pd.read_csv(C.METRICS_DIR / f"fold_scores_{task}_full.csv")
        fs = fs.drop(columns=["zero_r"], errors="ignore")
        fr = ST.friedman_nemenyi(fs)
        wil = ST.wilcoxon_vs_best(fs)
        wil.to_csv(C.METRICS_DIR / f"stat_wilcoxon_{task}.csv", index=False)
        table = pd.read_csv(C.METRICS_DIR / f"model_comparison_{task}_full.csv")
        # les deux meilleurs selon le critère de sélection (moyenne sur les plis communs)
        top2 = fs.mean().sort_values(ascending=False).index.tolist()[:2]
        from src.train import targets
        y = targets(test, task)
        X = feature_frame(test)
        p1 = _load(task, top2[0]).predict(X)
        p2 = _load(task, top2[1]).predict(X)
        mc = ST.mcnemar(y, p1, p2)
        mc.update({"model_a": top2[0], "model_b": top2[1]})
        res[task] = {"friedman": fr, "wilcoxon": wil.to_dict(orient="records"), "mcnemar": mc}
        print(f"[{task}] Friedman χ²={fr['chi2']:.2f} p={fr['p_value']:.2e} | "
              f"McNemar {top2[0]} vs {top2[1]} p={mc['p_value']:.3g}")
    return res


def feature_importance(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    reg = json.loads(C.REGISTRY_FILE.read_text(encoding="utf-8"))
    Xtr, ytr = feature_frame(train), train["label"].to_numpy()
    Xte, yte = feature_frame(test), test["label"].to_numpy()
    rf = _load("binary", "random_forest")
    xg = _load("binary", "xgboost")
    names = EX.transformed_names(rf)
    imp = pd.DataFrame(index=C.FEATURES + [f for f in ("total_bytes", "total_pkts", "byte_ratio", "pkt_ratio",
                                                      "bytes_per_pkt")])
    imp["impurity_rf"] = EX.aggregate_to_raw(names, rf.named_steps["clf"].feature_importances_)
    xnames = EX.transformed_names(xg)
    booster = xg.named_steps["clf"].get_booster()
    gain = booster.get_score(importance_type="gain")
    gvec = np.array([gain.get(f"f{i}", 0.0) for i in range(len(xnames))])
    imp["gain_xgboost"] = EX.aggregate_to_raw(xnames, gvec / gvec.sum())
    sub_te = test.sample(4000, random_state=RS)
    imp["shap_xgboost"] = EX.global_shap(xg, sub_te)
    best = reg["best"]["full"]["binary"]
    model = _load("binary", best)
    pi = permutation_importance(model, feature_frame(sub_te), sub_te["label"].to_numpy(),
                                scoring="f1", n_repeats=5, random_state=RS, n_jobs=2)
    imp.loc[C.FEATURES, "permutation_best"] = pi.importances_mean
    imp.loc[C.FEATURES, "permutation_std"] = pi.importances_std
    Z = rf.named_steps["prep"].transform(Xtr.iloc[:20000])
    mi = mutual_info_classif(Z, ytr[:20000], random_state=RS)
    imp["mutual_info"] = EX.aggregate_to_raw(names, mi)
    imp = imp.fillna(0.0)
    for c in ("impurity_rf", "gain_xgboost", "shap_xgboost", "permutation_best", "mutual_info"):
        imp[f"rank_{c}"] = imp[c].rank(ascending=False)
    imp["mean_rank"] = imp[[c for c in imp.columns if c.startswith("rank_")]].mean(axis=1)
    imp = imp.sort_values("mean_rank")
    imp.index.name = "feature"
    imp.to_csv(C.METRICS_DIR / "feature_importance.csv")
    # concordance des classements (Spearman entre méthodes)
    rk = imp[[c for c in imp.columns if c.startswith("rank_")]]
    rk.corr(method="spearman").to_csv(C.METRICS_DIR / "feature_importance_rank_agreement.csv")
    # SHAP multi-classe par catégorie (XGBoost)
    xm = _load("multiclass", "xgboost")
    vals, mnames, _ = EX.shap_values(xm, test.sample(2000, random_state=RS))
    per_cls = {}
    for ci, cname in enumerate(reg["class_names"]):
        per_cls[cname] = EX.aggregate_to_raw(mnames, np.abs(vals[:, ci, :]).mean(axis=0))
    pd.DataFrame(per_cls).to_csv(C.METRICS_DIR / "shap_multiclass_by_category.csv")
    return imp


def learning_curves(train: pd.DataFrame, use_cache: bool = True) -> dict:
    """Courbes d'apprentissage (F1, CV 5 plis) ; cache : reports/metrics/learning_curves.json."""
    cache = C.METRICS_DIR / "learning_curves.json"
    reg = json.loads(C.REGISTRY_FILE.read_text(encoding="utf-8"))
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("_best") == reg["best"]["full"]["binary"]:
            return {k: v for k, v in cached.items() if k != "_best"}
    out = {}
    # sous-échantillon stratifié de 20 000 flux : le SVM (O(n²)) resterait sinon des heures
    frac = min(1.0, 20000 / len(train))
    sub = pd.concat([g.sample(frac=frac, random_state=RS) for _, g in train.groupby("attack_cat")])
    X, y = feature_frame(sub), sub["label"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    for key in dict.fromkeys([reg["best"]["full"]["binary"], "decision_tree", "gaussian_nb"]):
        model = _load("binary", key)
        sizes, tr, va = learning_curve(clone(model), X, y, cv=cv, scoring="f1",
                                       train_sizes=np.linspace(0.05, 1.0, 7), n_jobs=2,
                                       shuffle=True, random_state=RS)
        out[key] = {"sizes": sizes.tolist(), "train_mean": tr.mean(1).tolist(),
                    "train_std": tr.std(1).tolist(), "val_mean": va.mean(1).tolist(),
                    "val_std": va.std(1).tolist()}
    cache.write_text(json.dumps({**out, "_best": reg["best"]["full"]["binary"]}), encoding="utf-8")
    return out


def ablation(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Retrait d'un groupe de features à la fois (XGBoost, hyperparamètres optimaux)."""
    from src.preprocessing import build_preprocessor
    from sklearn.pipeline import Pipeline
    xg = _load("binary", "xgboost")
    groups = {
        "aucun (référence)": [],
        "basic (Argus)": [c for c in C.FEATURE_GROUPS["basic"] if c in C.FEATURES],
        "content": [c for c in C.FEATURE_GROUPS["content"] if c in C.FEATURES],
        "time": [c for c in C.FEATURE_GROUPS["time"] if c in C.FEATURES],
        "additional general": C.FEATURE_GROUPS["additional_general"],
        "additional connection (ct_*)": C.FEATURE_GROUPS["additional_connection"],
        "TTL (sttl, dttl, ct_state_ttl)": C.TTL_FEATURES,
    }
    rows = []
    Xtr, ytr = feature_frame(train), train["label"].to_numpy()
    Xte, yte = feature_frame(test), test["label"].to_numpy()
    ym_tr = train["attack_cat"].map({c: i for i, c in enumerate(C.ATTACK_CATEGORIES)}).to_numpy()
    ym_te = test["attack_cat"].map({c: i for i, c in enumerate(C.ATTACK_CATEGORIES)}).to_numpy()
    xm = _load("multiclass", "xgboost")
    for g, cols in groups.items():
        r = {"removed_group": g, "n_removed": len(cols)}
        for task, base, yt, yv, avg in (("binary", xg, ytr, yte, "binary"),
                                         ("multiclass", xm, ym_tr, ym_te, "macro")):
            pipe = Pipeline([("prep", build_preprocessor(drop=cols)),
                             ("clf", clone(base.named_steps["clf"]))])
            pipe.fit(Xtr, yt)
            r[f"f1_{task}"] = f1_score(yv, pipe.predict(Xte), average=avg)
        rows.append(r)
        print(f"  ablation {g:<34} F1 bin={r['f1_binary']:.4f} | F1 macro={r['f1_multiclass']:.4f}")
    df = pd.DataFrame(rows)
    df["delta_binary"] = df["f1_binary"] - df.loc[0, "f1_binary"]
    df["delta_multiclass"] = df["f1_multiclass"] - df.loc[0, "f1_multiclass"]
    df.to_csv(C.METRICS_DIR / "ablation_feature_groups.csv", index=False)
    return df


def extractor_fidelity() -> pd.DataFrame:
    """Vérifie, sur 200 000 lignes du CSV officiel, que nos formules reproduisent les features."""
    from src.pcap.flows import ct_state_ttl
    df = clean_dataframe(read_raw_file(C.RAW_FILES[0], nrows=200000))
    rows = []

    def add(name, formula, ok):
        rows.append({"feature": name, "formule": formula, "concordance_pct": round(100 * float(np.mean(ok)), 3),
                     "n": int(len(ok))})
    d = df[(df.dur > 0) & (df.spkts > 1)]
    pred = (d.sbytes - d.sbytes / d.spkts) * 8 / d.dur
    add("sload", "(sbytes − sbytes/spkts)·8/dur", np.abs(pred - d.sload) <= 0.01 * d.sload.abs() + 1e-6)
    d2 = df[(df.dur > 0) & (df.dpkts > 1)]
    pred = (d2.dbytes - d2.dbytes / d2.dpkts) * 8 / d2.dur
    add("dload", "(dbytes − dbytes/dpkts)·8/dur", np.abs(pred - d2.dload) <= 0.01 * d2.dload.abs() + 1e-6)
    s = df[df.spkts > 0]
    add("smeansz", "round(sbytes/spkts)", np.abs(np.round(s.sbytes / s.spkts) - s.smeansz) <= 1)
    s = df[df.dpkts > 0]
    add("dmeansz", "round(dbytes/dpkts)", np.abs(np.round(s.dbytes / s.dpkts) - s.dmeansz) <= 1)
    add("tcprtt", "synack + ackdat", np.isclose(df.synack + df.ackdat, df.tcprtt, atol=1e-5))
    add("ct_state_ttl", "Algorithme 3.3 (règles TTL × état)",
        np.array([ct_state_ttl(a, b, st) for a, b, st in zip(df.sttl, df.dttl, df.state)]) == df.ct_state_ttl)
    add("is_sm_ips_ports", "Algorithme 3.2", ((df.srcip == df.dstip) & (df.sport == df.dsport)).astype(int)
        == df.is_sm_ips_ports)
    tcp = df[df.proto == "tcp"]
    add("swin (TCP)", "min(fenêtre du 1er segment, 255)", tcp.swin.isin([0, 255]))
    out = pd.DataFrame(rows)
    out.to_csv(C.METRICS_DIR / "extractor_fidelity.csv", index=False)
    print(out.to_string())
    return out


DEMO_PCAPS = {
    "demo_unsw_replay": "Rejeu d'enregistrements réels UNSW-NB15 (TTL d'origine)",
    "demo_unsw_replay_ttl64": "Même rejeu, TTL d'hôtes modernes (64)",
    "demo_unsw_testbed": "Scénarios synthétiques (topologie du banc de test)",
    "demo_modern_ttl": "Scénarios synthétiques (TTL 64/128)",
}


def pcap_end_to_end() -> list[dict]:
    """PCAP -> 47 features -> modèles déployés, mesuré sur la vérité terrain de chaque flux.

    Pour les rejeux, on rapporte aussi les métriques sur les seuls flux dont
    l'enregistrement d'origine n'a jamais été vu à l'entraînement.
    """
    from src.models import predict_flows
    from src.pcap import pcap_to_flows
    from src.pcap.generator import match_ground_truth
    from src.evaluate import binary_metrics
    rows = []
    for name, desc in DEMO_PCAPS.items():
        pcap = C.SAMPLE_PCAP_DIR / f"{name}.pcap"
        if not pcap.exists():
            continue
        lab = pd.read_csv(C.SAMPLE_PCAP_DIR / f"{name}_labels.csv")
        t0 = time.time()
        flows = pcap_to_flows(pcap)
        t_ext = time.time() - t0
        m = match_ground_truth(flows, lab)
        for profile in ("full", "portable"):
            pr = predict_flows(flows, profile=profile)
            ok_all = m["true_label"].notna().to_numpy()
            subsets = [("tous les flux", ok_all)]
            if "in_train" in lab and lab["in_train"].any():
                subsets.append(("jamais vus à l'entraînement", ok_all & (m["true_in_train"] == False).to_numpy()))  # noqa: E712
            for sub, ok in subsets:
                y, yp = m["true_label"][ok].astype(int).to_numpy(), pr["prediction"][ok].to_numpy()
                bm = binary_metrics(y, yp, pr["proba_attaque"][ok].to_numpy())
                att = (y == 1)
                cat_acc = float((pr["categorie_predite"][ok].to_numpy()[att] ==
                                 m["true_attack_cat"][ok].to_numpy()[att]).mean()) if att.any() else float("nan")
                rows.append({"pcap": name, "description": desc, "profile": profile, "subset": sub,
                             "n_packets": int(flows.attrs.get("n_packets", 0)), "n_flows": int(ok.sum()),
                             "n_attacks": int(att.sum()), "extraction_s": round(t_ext, 3),
                             "accuracy": bm["accuracy"], "precision": bm["precision"], "recall": bm["recall"],
                             "f1": bm["f1"], "false_alarm_rate": bm["false_alarm_rate"],
                             "attack_type_accuracy": cat_acc})
                print(f"  {name:<24} {profile:<8} {sub:<28} n={ok.sum():5d} F1={bm['f1']:.3f} "
                      f"rappel={bm['recall']:.3f} FPR={bm['false_alarm_rate']:.3f} type={cat_acc:.3f}")
    pd.DataFrame(rows).to_csv(C.METRICS_DIR / "pcap_end_to_end.csv", index=False)
    return rows


def replay_fidelity() -> list[dict]:
    """Test aller-retour : enregistrement réel -> PCAP rejoué -> features ré-extraites."""
    from src.pcap import pcap_to_flows
    from src.pcap.replay import roundtrip_fidelity
    rec_path = C.SAMPLE_PCAP_DIR / "demo_unsw_replay_records.csv"
    if not rec_path.exists():
        return []
    records = pd.read_csv(rec_path, low_memory=False)
    flows = pcap_to_flows(C.SAMPLE_PCAP_DIR / "demo_unsw_replay.pcap")
    fid = roundtrip_fidelity(records, flows)
    fid["n_records"], fid["n_flows"] = len(records), len(flows)
    fid.to_csv(C.METRICS_DIR / "replay_fidelity.csv", index=False)
    print(fid.to_string())
    return fid.to_dict(orient="records")


def main():
    t0 = time.time()
    train = pd.read_csv(C.TRAIN_FILE, low_memory=False)
    test = pd.read_csv(C.TEST_FILE, low_memory=False)
    out = {}
    print("1) Statistiques EDA");             out["eda"] = eda_statistics(train)
    print("2) Comparaison statistique");      out["model_stats"] = model_statistics(test)
    print("3) Importance des features");      imp = feature_importance(train, test)
    out["feature_importance"] = imp.reset_index().head(45).to_dict(orient="records")
    print("4) Courbes d'apprentissage");      out["learning_curves"] = learning_curves(train)
    print("5) Ablation");                     out["ablation"] = ablation(train, test).to_dict(orient="records")
    print("6) Fidélité extracteur");          out["fidelity"] = extractor_fidelity().to_dict(orient="records")
    print("7) PCAP bout-en-bout");            out["pcap_e2e"] = pcap_end_to_end()
    print("8) Fidélité aller-retour (rejeu)"); out["replay_fidelity"] = replay_fidelity()
    out["elapsed_s"] = round(time.time() - t0, 1)
    (C.METRICS_DIR / "analysis.json").write_text(json.dumps(out, default=float, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"Terminé en {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
