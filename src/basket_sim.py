"""
Synthetic shopping baskets with PLANTED correlations (ground truth), plus a
detection-error model that mimics what the vision pipeline gets wrong.

Why: a 3x3 shelf can't produce thousands of real baskets, but Apriori needs
them. Planting known rules lets us measure how many of them Apriori
recovers - first from clean baskets, then from baskets passed through
realistic YOLO errors (missed items, spurious items). That comparison is the
link between the IVP half and the AI half of the paper.
"""
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class PlantedRule:
    antecedent: tuple   # items that trigger the rule
    consequent: str     # item that becomes more likely
    strength: float     # P(consequent is added | antecedent all present)

    def itemset(self):
        return frozenset(self.antecedent) | {self.consequent}

    def __str__(self):
        return f"{' + '.join(self.antecedent)} -> {self.consequent}  (p={self.strength})"


def default_planted_rules(products):
    """
    Default rules by POSITION in the product list, so they work whatever the
    real class names are. Mix of 'obvious' and 'impossible' (cross-category)
    pairings, in the spirit of beers-and-diapers.
    """
    p = products
    return [
        PlantedRule((p[0],), p[1], 0.55),
        PlantedRule((p[2],), p[5], 0.50),
        PlantedRule((p[3], p[4]), p[7], 0.60),
        PlantedRule((p[6],), p[9], 0.45),
    ]


def simulate_baskets(products, n_baskets=2000, planted=None, base_range=(0.04, 0.20), seed=42):
    """
    Each product appears independently with its own base probability; then
    each planted rule fires (adds its consequent) with probability `strength`
    when its antecedent is present. Empty baskets are resampled.
    Returns (baskets: list[list[str]], base_probs: dict).
    """
    rng = np.random.default_rng(seed)
    planted = planted if planted is not None else default_planted_rules(products)
    base = dict(zip(products, rng.uniform(*base_range, size=len(products))))

    baskets = []
    while len(baskets) < n_baskets:
        basket = {item for item in products if rng.random() < base[item]}
        for rule in planted:
            if all(a in basket for a in rule.antecedent) and rng.random() < rule.strength:
                basket.add(rule.consequent)
        if basket:
            baskets.append(sorted(basket))
    return baskets, base


# ----------------------------------------------------------------------------
# Detection-error model
# ----------------------------------------------------------------------------
def apply_detection_noise(baskets, products, recall=0.9, fp_rate=0.05, swap_rate=0.0, seed=0):
    """
    Corrupt baskets the way an imperfect detector would:
      recall    - P(a removed item is detected). float, or {product: recall}
      swap_rate - P(a detected item is given the WRONG class)
      fp_rate   - P(a basket gains one spurious item that wasn't taken)
    """
    rng = np.random.default_rng(seed)
    recall_of = recall if isinstance(recall, dict) else {p: recall for p in products}
    noisy = []
    for basket in baskets:
        out = set()
        for item in basket:
            if rng.random() > recall_of.get(item, 1.0):
                continue  # missed
            if swap_rate and rng.random() < swap_rate:
                item = rng.choice([p for p in products if p != item])
            out.add(str(item))
        if rng.random() < fp_rate:
            absent = [p for p in products if p not in out]
            if absent:
                out.add(str(rng.choice(absent)))
        if out:
            noisy.append(sorted(out))
    return noisy


def noise_profile_from_yolo(results_json):
    """
    Build a noise profile from notebook 02's saved metrics:
      per-class recall      -> P(detected)
      mean (1 - precision)  -> fp_rate (approximation: chance a basket picks
                               up one wrong item)
    """
    with open(results_json) as f:
        res = json.load(f)
    per_class = res["per_class"]
    recall = {name: float(v["recall"]) for name, v in per_class.items()}
    fp_rate = float(np.mean([1 - v["precision"] for v in per_class.values()]))
    if fp_rate > 0.5:
        print(f"WARNING: detector precision is very low (fp_rate={fp_rate:.2f}); capping at 0.5. "
              "Is this an under-trained / demo model?")
        fp_rate = 0.5
    return {"recall": recall, "fp_rate": fp_rate}
