"""Confidence-aware post-processing: evidence gating + one-hop graph expansion.

Both are OPT-IN (flags default off) so the baseline path is untouched.

evidence_gate
    Two-stage decision per Source 1 row. Stage 1 asks "is there a reliable
    match?" -- if the best candidate probability clears ``gate_high`` the
    row's thresholded keeps are accepted as-is. Otherwise (best prob in
    [threshold, gate_high)) every keep must carry independent corroborating
    evidence (postcode agreement, or strong name + country agreement);
    without it the row is emitted as a singleton. This directly targets the
    macro-F0.5 singleton sensitivity: a wrong merge on a true singleton
    scores 0.0 for that row.

graph expansion (one hop, evidence-gated -- NOT blind transitive closure)
    Implemented in ``pipeline.score_chunk``: for a query row with an
    accepted match above ``graph_leg_thr``, the matched *target's own*
    blocking keys are looked up to find further targets similar to it; any
    new target is scored directly against the query with the full feature
    vector and accepted only if it clears the model threshold. A transitive
    link is therefore created only when the direct pairwise evidence
    supports it -- the graph proposes, the classifier disposes.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def evidence_gate(
    keep_ids: list[str],
    probs: list[float],
    X: np.ndarray,
    *,
    threshold: float,
    gate_high: float = 0.90,
) -> list[str]:
    """Filter one row's thresholded keeps through the two-stage rule.

    ``keep_ids``/``probs``/``X`` are aligned (same order). Returns the
    subset to emit. Pure function -- unit-testable without stores.
    """
    if not keep_ids:
        return []
    best = max(probs)
    if best >= gate_high:
        return list(keep_ids)
    pc = _IDX["postcode_match"]
    nr = _IDX["name_ratio"]
    cm = _IDX["country_match"]
    out = []
    for cid, p, row in zip(keep_ids, probs, X):
        if p < threshold:
            continue
        corroborated = bool(
            row[pc] >= 1.0 or (row[nr] >= 0.85 and row[cm] >= 1.0)
        )
        if corroborated:
            out.append(cid)
    return out
