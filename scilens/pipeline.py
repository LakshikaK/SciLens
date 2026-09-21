from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from scilens.classifier import classify_papers
from scilens.consensus import aggregate_evidence
from scilens.pubmed import retrieve_papers
from scilens.retriever import filter_relevant, rank_papers


@dataclass
class AnalysisResult:
    question: str
    n_retrieved: int
    n_relevant: int
    verdict: str
    confidence: float
    counts: dict[str, int]
    weighted_shares: dict[str, float]
    explanation: str
    papers: list[dict] = field(default_factory=list)
    discarded: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _paper_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    columns = [
        col
        for col in (
            "pmid",
            "title",
            "abstract",
            "year",
            "journal",
            "url",
            "similarity",
            "label",
            "confidence",
        )
        if col in df.columns
    ]
    records = df[columns].to_dict(orient="records")
    for row in records:
        if "similarity" in row and row["similarity"] is not None:
            row["similarity"] = round(float(row["similarity"]), 4)
        if "confidence" in row and row["confidence"] is not None:
            row["confidence"] = round(float(row["confidence"]), 4)
    return records


def analyze_question(
    question: str,
    max_results: int = 20,
    min_similarity: float = 0.35,
    backend: str = "nli",
    papers: pd.DataFrame | None = None,
    classifier=None,
    embedder=None,
) -> AnalysisResult:
    """Run PubMed retrieval → embedding rank → filter → classify → consensus."""
    claim = question.strip()
    if not claim:
        raise ValueError("Question cannot be empty.")

    retrieved = papers if papers is not None else retrieve_papers(claim, max_results=max_results)
    ranked = rank_papers(claim, retrieved, embedder=embedder)
    relevant = filter_relevant(ranked, min_similarity=min_similarity)
    discarded = (
        ranked.loc[ranked["similarity"] < min_similarity]
        if not ranked.empty and "similarity" in ranked.columns
        else ranked.iloc[0:0]
    )
    classified = classify_papers(claim, relevant, backend=backend, classifier=classifier)
    consensus = aggregate_evidence(classified)

    return AnalysisResult(
        question=claim,
        n_retrieved=len(retrieved),
        n_relevant=len(classified),
        verdict=consensus["verdict"],
        confidence=float(consensus["confidence"]),
        counts=consensus["counts"],
        weighted_shares=consensus["weighted_shares"],
        explanation=consensus["explanation"],
        papers=_paper_records(classified),
        discarded=_paper_records(discarded.head(12)),
    )
