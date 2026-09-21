from __future__ import annotations

import pandas as pd

LABELS = ("Supports", "Contradicts", "Mixed")


def _empty_counts() -> dict[str, int]:
    return {label: 0 for label in LABELS}


def aggregate_evidence(papers: pd.DataFrame) -> dict:
    """Turn classified papers into a consensus verdict, confidence, and explanation."""
    if papers.empty:
        return {
            "verdict": "Insufficient evidence",
            "confidence": 0.0,
            "counts": _empty_counts(),
            "weighted_shares": {label: 0.0 for label in LABELS},
            "explanation": (
                "No papers passed the relevance filter, so SciLens will not force a conclusion."
            ),
        }

    working = papers.copy()
    working["label"] = working["label"].where(working["label"].isin(LABELS), "Mixed")
    working["confidence"] = pd.to_numeric(working.get("confidence"), errors="coerce").fillna(0.5)
    working["similarity"] = pd.to_numeric(working.get("similarity"), errors="coerce").fillna(0.0)
    working["weight"] = (working["similarity"].clip(lower=0) * working["confidence"].clip(lower=0.05)).clip(lower=0.01)

    counts = _empty_counts()
    counts.update(working["label"].value_counts().to_dict())
    weighted = working.groupby("label")["weight"].sum()
    total_weight = float(working["weight"].sum()) or 1.0
    shares = {label: float(weighted.get(label, 0.0)) / total_weight for label in LABELS}

    support = shares["Supports"]
    contradict = shares["Contradicts"]
    mixed = shares["Mixed"]
    n = len(working)

    if support >= 0.55 and support > contradict + 0.1:
        verdict = "Supports"
        lean = "the available relevant papers mostly support the claim"
    elif contradict >= 0.55 and contradict > support + 0.1:
        verdict = "Contradicts"
        lean = "the available relevant papers mostly contradict the claim"
    else:
        verdict = "Mixed evidence"
        lean = "relevant papers disagree or report mixed findings"

    polarity = abs(support - contradict)
    sample_factor = min(1.0, n / 8.0)
    mean_sim = float(working["similarity"].mean())
    confidence = round(min(0.95, (0.35 + 0.4 * polarity + 0.15 * (1 - mixed) + 0.1 * mean_sim) * (0.55 + 0.45 * sample_factor)), 3)

    explanation = (
        f"{n} relevant paper(s) were retained after semantic filtering. "
        f"Weighted evidence: {support:.0%} support, {contradict:.0%} contradict, {mixed:.0%} mixed. "
        f"Overall, {lean}. Confidence reflects agreement, relevance scores, and sample size — not medical certainty."
    )
    return {
        "verdict": verdict,
        "confidence": confidence,
        "counts": counts,
        "weighted_shares": shares,
        "explanation": explanation,
    }
