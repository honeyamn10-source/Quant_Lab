"""Synthetic market generators with known ground truth (hypothesis tests)."""

from quantlab.synthetic.markets import available_processes, generate, generate_pairs

__all__ = ["available_processes", "generate", "generate_pairs", "generate_universe"]


def generate_universe(processes: dict[str, dict], seed: int = 1) -> dict[str, object]:
    """Build a multi-symbol synthetic universe.

    processes maps symbol -> {"process": ..., "n": ..., "seed": ...}.
    Returns dict[symbol, BarSeries].
    """
    out: dict[str, object] = {}
    for symbol, params in processes.items():
        out[symbol] = generate(
            params.get("process", "random_walk"),
            n=int(params.get("n", 504)),
            seed=int(params.get("seed", seed)),
            symbol=symbol,
        )
    return out
