"""
AI half: Apriori association rules + KNN next-item recommendation, with the
metrics reported in the paper (rule recovery, Hit@N).
"""
import inspect

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------
def to_matrix(baskets, items=None):
    """List of baskets -> boolean one-hot DataFrame (rows = baskets, cols = items)."""
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    if items is not None:  # fixed column order, include never-seen items
        df = df.reindex(columns=list(items), fill_value=False)
    return df.astype(bool)


# ----------------------------------------------------------------------------
# Apriori
# ----------------------------------------------------------------------------
def mine_rules(matrix, min_support=0.02, min_lift=1.0, min_confidence=0.0, max_len=3):
    """Frequent itemsets (Apriori) -> association rules filtered by lift & confidence."""
    itemsets = apriori(matrix, min_support=min_support, use_colnames=True, max_len=max_len)
    if itemsets.empty:
        return itemsets, pd.DataFrame()
    kwargs = {"metric": "lift", "min_threshold": min_lift}
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kwargs["num_itemsets"] = len(matrix)  # required by mlxtend >= 0.23.2
    rules = association_rules(itemsets, **kwargs)
    rules = rules[rules["confidence"] >= min_confidence]
    return itemsets, rules.sort_values("lift", ascending=False).reset_index(drop=True)


def readable(rules, n=10):
    """Top-n rules as a tidy table with string item names."""
    if rules.empty:
        return rules
    out = rules.head(n)[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return out.round(3).reset_index(drop=True)


def rule_recovery(rules, planted, top_k=20):
    """
    recall      - fraction of planted rules whose full itemset appears as a mined rule
    precision@k - among the top-k DISTINCT itemsets (by lift), fraction that
                  contain a planted "signal pair" {antecedent item, consequent}.
                  Itemsets with no planted pair are spurious correlations.
    """
    planted_sets = [r.itemset() for r in planted]
    signal_pairs = [frozenset((a, r.consequent)) for r in planted for a in r.antecedent]
    if rules.empty:
        return {"recall": 0.0, "precision_at_k": 0.0, "recovered": [], "k": top_k}
    mined_sets = []
    for a, c in zip(rules["antecedents"], rules["consequents"]):
        s = frozenset(a) | frozenset(c)
        if s not in mined_sets:
            mined_sets.append(s)
    recovered = [str(r) for r, ps in zip(planted, planted_sets) if ps in mined_sets]
    top = mined_sets[:top_k]
    true_top = sum(any(pair <= s for pair in signal_pairs) for s in top)
    return {
        "recall": len(recovered) / len(planted),
        "precision_at_k": true_top / len(top) if top else 0.0,
        "recovered": recovered,
        "k": top_k,
    }


# ----------------------------------------------------------------------------
# KNN next-item recommendation
# ----------------------------------------------------------------------------
def fit_knn(matrix, n_neighbors=5, metric="jaccard"):
    return NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(matrix.values.astype(bool))


def recommend(knn, train_matrix, basket_items, top_n=3):
    """Items most common in the neighbours' baskets that the customer doesn't have yet."""
    cols = list(train_matrix.columns)
    q = np.zeros((1, len(cols)), dtype=bool)
    for item in basket_items:
        q[0, cols.index(item)] = True
    dist, idx = knn.kneighbors(q)
    votes = train_matrix.values[idx[0]].sum(axis=0).astype(float)
    votes += train_matrix.values.mean(axis=0) * 1e-3  # tie-break by popularity
    votes[q[0]] = -1
    ranked = [cols[i] for i in np.argsort(-votes)[:top_n]]
    neighbours = [[cols[j] for j in np.flatnonzero(train_matrix.values[i])] for i in idx[0]]
    return ranked, neighbours, dist[0]


def _hide_one(test_matrix, seed):
    """For each test basket with >= 2 items, hide one random item."""
    rng = np.random.default_rng(seed)
    X = test_matrix.values.astype(bool).copy()
    rows, hidden = [], []
    for i in range(len(X)):
        present = np.flatnonzero(X[i])
        if len(present) < 2:
            continue
        h = rng.choice(present)
        X[i, h] = False
        rows.append(i)
        hidden.append(h)
    return X[rows], np.array(hidden)


def knn_hit_rate(train_matrix, test_matrix, n_neighbors=5, metric="jaccard", top_n=3, seed=0):
    """Leave-one-out next-item prediction: Hit@N = P(hidden item in top-N recommendations)."""
    Q, hidden = _hide_one(test_matrix, seed)
    if len(Q) == 0:
        return float("nan")
    train = train_matrix.values.astype(bool)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(train)
    _, idx = knn.kneighbors(Q)
    votes = train[idx].sum(axis=1).astype(float)          # (n_queries, n_items)
    votes += train.mean(axis=0) * 1e-3                    # popularity tie-break
    votes[Q] = -1                                         # never recommend what's in the cart
    top = np.argsort(-votes, axis=1)[:, :top_n]
    return float(np.mean([h in t for h, t in zip(hidden, top)]))


def popularity_hit_rate(train_matrix, test_matrix, top_n=3, seed=0):
    """Baseline: always recommend the globally most popular items not already in the cart."""
    Q, hidden = _hide_one(test_matrix, seed)
    if len(Q) == 0:
        return float("nan")
    pop = train_matrix.values.mean(axis=0)
    votes = np.tile(pop, (len(Q), 1))
    votes[Q] = -1
    top = np.argsort(-votes, axis=1)[:, :top_n]
    return float(np.mean([h in t for h, t in zip(hidden, top)]))
