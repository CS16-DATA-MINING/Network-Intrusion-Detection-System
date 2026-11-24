"""Règles d'association (Apriori) sur des flux réseau discrétisés.

Chaque flux devient une transaction d'items « attribut=modalité » (variables
numériques discrétisées par quantiles sur l'échantillon). Apriori (Agrawal &
Srikant, 1994) énumère les itemsets fréquents niveau par niveau grâce à la
propriété d'anti-monotonie du support, puis on génère les règles X ⇒ Y avec
support, confiance, lift, levier et conviction. On s'intéresse en particulier
aux règles dont le conséquent est la classe (``classe=Attaque`` ou une
catégorie), qui donnent des signatures interprétables d'intrusion.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

DEFAULT_NUMERIC = ["sbytes", "dbytes", "dur", "sttl", "dttl", "smeansz", "dmeansz",
                   "ct_srv_src", "ct_dst_src_ltm", "sload", "spkts"]
DEFAULT_CATEG = ["proto", "state", "service"]


def discretize(df: pd.DataFrame, numeric=DEFAULT_NUMERIC, categ=DEFAULT_CATEG,
               bins: int = 3, target: str | None = "label") -> list[frozenset]:
    """Transforme les flux en transactions (ensembles d'items)."""
    cols = {}
    labels = {3: ["bas", "moyen", "élevé"], 4: ["Q1", "Q2", "Q3", "Q4"]}.get(bins)
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.nunique() <= bins + 1:
            cols[c] = c + "=" + x.astype(int).astype(str)
            continue
        q = pd.qcut(x.rank(method="first"), bins, labels=labels)
        cols[c] = c + "=" + q.astype(str)
    for c in categ:
        top = df[c].value_counts().index[:6]
        cols[c] = c + "=" + df[c].where(df[c].isin(top), "autre").astype(str)
    if target == "label":
        cols["classe"] = "classe=" + np.where(df["label"] == 1, "Attaque", "Normal")
    elif target:
        cols["classe"] = "classe=" + df[target].astype(str)
    items = pd.DataFrame(cols)
    return [frozenset(r) for r in items.to_numpy()]


def apriori(transactions: list[frozenset], min_support: float = 0.05,
            max_len: int = 4) -> dict[frozenset, float]:
    """Itemsets fréquents (support >= min_support) par l'algorithme Apriori."""
    n = len(transactions)
    min_count = min_support * n
    counts: dict[frozenset, int] = {}
    for t in transactions:
        for it in t:
            s = frozenset([it])
            counts[s] = counts.get(s, 0) + 1
    current = {s: c for s, c in counts.items() if c >= min_count}
    frequent = dict(current)
    k = 2
    while current and k <= max_len:
        # génération des candidats (jointure) + élagage par anti-monotonie
        prev = list(current)
        prev_set = set(prev)
        cands = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                u = prev[i] | prev[j]
                if len(u) == k and all(frozenset(s) in prev_set for s in combinations(u, k - 1)):
                    cands.add(u)
        if not cands:
            break
        cnt = dict.fromkeys(cands, 0)
        for t in transactions:
            if len(t) < k:
                continue
            for c in cands:
                if c <= t:
                    cnt[c] += 1
        current = {s: c for s, c in cnt.items() if c >= min_count}
        frequent.update(current)
        k += 1
    return {s: c / n for s, c in frequent.items()}


def association_rules(freq: dict[frozenset, float], min_confidence: float = 0.6,
                      consequent_prefix: str | None = None) -> pd.DataFrame:
    rows = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for cons in combinations(itemset, r):
                cons = frozenset(cons)
                if consequent_prefix and not all(i.startswith(consequent_prefix) for i in cons):
                    continue
                ant = itemset - cons
                if ant not in freq or cons not in freq:
                    continue
                conf = sup / freq[ant]
                if conf < min_confidence:
                    continue
                lift = conf / freq[cons]
                lev = sup - freq[ant] * freq[cons]
                conv = (1 - freq[cons]) / (1 - conf) if conf < 1 else np.inf
                rows.append({"antecedent": " ∧ ".join(sorted(ant)),
                             "consequent": " ∧ ".join(sorted(cons)),
                             "support": sup, "confidence": conf, "lift": lift,
                             "leverage": lev, "conviction": conv, "length": len(itemset)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
