"""SAASBO glue — sparse axis-aligned subspace BO for large search spaces (#18).

SAASBO uses sparsity priors + fully-Bayesian (NUTS) inference to stay sample-efficient in
high dimensions. It's slow (fully Bayesian), so it's gated behind a tunable ``min_dims``.

Complements dimensionality reduction (#17): SAASBO *models* high-dim directly; A1 *prunes*
dead dims at fixed points. Use either, or both (A1 shrinks, then a smaller GP suffices).

The wiring is just a surrogate swap on the existing BoTorch node — same acquisition logic.
"""

from __future__ import annotations

import logging

log = logging.getLogger("axbo_extensions.saasbo")


def should_use_saasbo(n_params: int, cfg) -> bool:
    """True iff SAASBO is enabled and the dimensionality meets the tunable threshold."""
    return bool(cfg and getattr(cfg, "enabled", False) and n_params >= cfg.min_dims)


def saasbo_model_class(n_params: int, cfg):
    """``SaasFullyBayesianSingleTaskGP`` if SAASBO should engage at this dimensionality, else
    ``None`` (caller falls back to its default surrogate). Ceiling: NUTS fit is slow — keep
    ``min_dims`` high enough that the sparsity payoff covers the cost."""
    if not should_use_saasbo(n_params, cfg):
        return None
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    log.info("SAASBO engaged: %d params >= min_dims=%d", n_params, cfg.min_dims)
    return SaasFullyBayesianSingleTaskGP
