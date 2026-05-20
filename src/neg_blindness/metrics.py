from __future__ import annotations

import random
from statistics import mean


def bootstrap_ci(
    values: list[float | bool],
    n_samples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    numeric = [float(value) for value in values]
    rng = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(n_samples):
        sample = [numeric[rng.randrange(len(numeric))] for _ in range(len(numeric))]
        estimates.append(mean(sample))
    estimates.sort()
    lower_idx = int(((1.0 - confidence) / 2.0) * len(estimates))
    upper_idx = int((1.0 - (1.0 - confidence) / 2.0) * len(estimates)) - 1
    return {
        "mean": mean(numeric),
        "lower": estimates[max(0, lower_idx)],
        "upper": estimates[min(len(estimates) - 1, upper_idx)],
    }

