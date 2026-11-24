"""Apprentissage non supervisé : clustering et détection d'anomalies.

Algorithmes du cours (chapitre 4) : K-Means (méthode du coude, k-means++),
K-Medoids (PAM : phases BUILD + SWAP, et CLARA pour passer à l'échelle),
AGNES (liaisons Ward, complète, moyenne, simple), DIANA (divisif), DBSCAN
(graphe des k-distances). Extensions : modèles de mélange gaussiens (GMM, BIC)
et détection d'intrusions non supervisée (Isolation Forest, LOF, distance au
centroïde K-Means).

PAM et DIANA sont implémentés ici (ils n'existent pas dans scikit-learn).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, homogeneity_score,
                             normalized_mutual_info_score, rand_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

RS = 42


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def purity(y_true, labels) -> float:
    ct = pd.crosstab(labels, y_true)
    return float(ct.max(axis=1).sum() / ct.to_numpy().sum())


def internal_metrics(X, labels, sample: int = 4000) -> dict:
    mask = labels >= 0                       # ignore le bruit DBSCAN (-1)
    Xm, lm = X[mask], labels[mask]
    if len(np.unique(lm)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if len(Xm) > sample:
        idx = np.random.default_rng(RS).choice(len(Xm), sample, replace=False)
        sil = silhouette_score(Xm[idx], lm[idx])
    else:
        sil = silhouette_score(Xm, lm)
    return {"silhouette": float(sil), "davies_bouldin": float(davies_bouldin_score(Xm, lm)),
            "calinski_harabasz": float(calinski_harabasz_score(Xm, lm))}


def external_metrics(y_true, labels) -> dict:
    return {"rand_index": float(rand_score(y_true, labels)),
            "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
            "nmi": float(normalized_mutual_info_score(y_true, labels)),
            "homogeneity": float(homogeneity_score(y_true, labels)),
            "purity": purity(y_true, labels)}


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def kmeans_scan(X, ks=range(2, 13), init="k-means++") -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, init=init, n_init=5, random_state=RS).fit(X)
        r = {"k": k, "init": init, "inertia_wcss": float(km.inertia_), "n_iter": int(km.n_iter_)}
        r.update(internal_metrics(X, km.labels_))
        rows.append(r)
    return pd.DataFrame(rows)


def elbow_k(inertia: np.ndarray, ks) -> int:
    """Coude = point le plus éloigné de la droite reliant les extrémités (méthode « kneedle »)."""
    ks = np.asarray(list(ks), dtype=float)
    y = np.asarray(inertia, dtype=float)
    xn, yn = (ks - ks[0]) / (ks[-1] - ks[0]), (y - y[-1]) / (y[0] - y[-1])
    dist = np.abs(yn - (1 - xn))
    return int(ks[np.argmax(dist)])


# ---------------------------------------------------------------------------
# K-Medoids : PAM (BUILD + SWAP) et CLARA
# ---------------------------------------------------------------------------
def pam(X=None, k=4, D=None, max_iter=100, metric="euclidean"):
    """Partitioning Around Medoids (Kaufman & Rousseeuw).

    BUILD : choix glouton des medoïdes ; SWAP : échange medoïde / non-medoïde
    tant que le coût total (somme des dissimilarités au medoïde) diminue.
    Retourne (indices des medoïdes, étiquettes, coût).
    """
    if D is None:
        D = squareform(pdist(X, metric=metric))
    n = D.shape[0]
    # BUILD
    medoids = [int(np.argmin(D.sum(axis=1)))]
    nearest = D[:, medoids[0]].copy()
    for _ in range(1, k):
        gains = np.maximum(nearest[:, None] - D, 0).sum(axis=0)
        gains[medoids] = -1
        m = int(np.argmax(gains))
        medoids.append(m)
        nearest = np.minimum(nearest, D[:, m])
    medoids = np.array(medoids)
    # SWAP
    cost = D[:, medoids].min(axis=1).sum()
    for _ in range(max_iter):
        improved = False
        dm = D[:, medoids]
        order = np.argsort(dm, axis=1)
        d1 = dm[np.arange(n), order[:, 0]]
        d2 = dm[np.arange(n), order[:, 1]] if k > 1 else np.full(n, np.inf)
        best_delta, best_swap = 0.0, None
        non_med = np.setdiff1d(np.arange(n), medoids)
        for i, m in enumerate(medoids):
            in_cluster = order[:, 0] == i
            # coût si m est remplacé par chaque candidat h (vectorisé sur h)
            Dh = D[:, non_med]
            new_d = np.where(in_cluster[:, None], np.minimum(d2[:, None], Dh),
                             np.minimum(d1[:, None], Dh))
            delta = new_d.sum(axis=0) - d1.sum()
            j = int(np.argmin(delta))
            if delta[j] < best_delta - 1e-9:
                best_delta, best_swap = float(delta[j]), (i, int(non_med[j]))
        if best_swap is not None:
            medoids[best_swap[0]] = best_swap[1]
            cost += best_delta
            improved = True
        if not improved:
            break
    labels = D[:, medoids].argmin(axis=1)
    return medoids, labels, float(cost)


def clara(X, k=4, n_samples=5, sample_size=800, seed=RS):
    """CLARA : PAM sur plusieurs échantillons, on garde les medoïdes de coût global minimal."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_samples):
        idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
        med, _, _ = pam(X[idx], k)
        centers = X[idx[med]]
        d = cdist(X, centers)
        cost = d.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, centers, d.argmin(axis=1))
    return best[1], best[2], float(best[0])


# ---------------------------------------------------------------------------
# Hiérarchique : AGNES et DIANA
# ---------------------------------------------------------------------------
def agnes(X, methods=("ward", "complete", "average", "single"), k=4) -> tuple[pd.DataFrame, dict]:
    rows, links = [], {}
    d = pdist(X)
    for m in methods:
        Z = linkage(X, method=m) if m == "ward" else linkage(d, method=m)
        links[m] = Z
        coph, _ = cophenet(Z, d)
        lab = fcluster(Z, k, criterion="maxclust") - 1
        r = {"linkage": m, "k": k, "cophenetic_corr": float(coph),
             "largest_cluster_pct": float(np.bincount(lab).max() / len(lab) * 100)}
        r.update(internal_metrics(X, lab))
        rows.append(r)
    return pd.DataFrame(rows), links


def diana(X, k=4):
    """DIANA (DIvisive ANAlysis, Kaufman & Rousseeuw) : approche descendante.

    À chaque étape on scinde le cluster de plus grand diamètre : l'objet le plus
    « mécontent » (dissimilarité moyenne maximale) initie un groupe dissident,
    puis on y transfère les objets plus proches du dissident que du reste.
    Retourne (étiquettes, historique des scissions).
    """
    D = squareform(pdist(X))
    clusters = [np.arange(len(X))]
    history = []
    while len(clusters) < k:
        diam = [D[np.ix_(c, c)].max() if len(c) > 1 else 0 for c in clusters]
        ci = int(np.argmax(diam))
        c = clusters.pop(ci)
        sub = D[np.ix_(c, c)]
        avg = sub.sum(axis=1) / (len(c) - 1)
        splinter = [int(np.argmax(avg))]
        rest = [i for i in range(len(c)) if i != splinter[0]]
        moved = True
        while moved and len(rest) > 1:
            moved = False
            d_rest = np.array([sub[i, [j for j in rest if j != i]].mean() for i in rest])
            d_spl = np.array([sub[i, splinter].mean() for i in rest])
            diff = d_rest - d_spl
            j = int(np.argmax(diff))
            if diff[j] > 0:
                splinter.append(rest.pop(j))
                moved = True
        a, b = c[np.array(splinter)], c[np.array(rest)]
        history.append({"split_cluster_size": int(len(c)), "diameter": float(diam[ci]),
                        "sizes": [int(len(a)), int(len(b))]})
        clusters += [a, b]
    labels = np.empty(len(X), dtype=int)
    for i, c in enumerate(clusters):
        labels[c] = i
    return labels, history


# ---------------------------------------------------------------------------
# DBSCAN, GMM
# ---------------------------------------------------------------------------
def k_distance(X, k=10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, -1])


def dbscan_scan(X, eps_values, min_samples_values=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    for eps in eps_values:
        for ms in min_samples_values:
            lab = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(X)
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            r = {"eps": float(eps), "min_samples": ms, "n_clusters": n_cl,
                 "noise_pct": float((lab == -1).mean() * 100)}
            r.update(internal_metrics(X, lab) if n_cl >= 2 else
                     {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan})
            rows.append(r)
    return pd.DataFrame(rows)


def gmm_scan(X, ks=range(2, 13)) -> pd.DataFrame:
    rows = []
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=RS,
                            reg_covar=1e-4, max_iter=300).fit(X)
        lab = g.predict(X)
        r = {"k": k, "bic": float(g.bic(X)), "aic": float(g.aic(X))}
        r.update(internal_metrics(X, lab))
        rows.append(r)
    return pd.DataFrame(rows)
