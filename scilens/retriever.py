from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_model: Any = None


class Embedder(Protocol):
    def encode(self, texts: str | list[str], **kwargs: Any) -> Any: ...


def get_embedder() -> Embedder:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _model


def paper_text(row: pd.Series) -> str:
    return f"{row.get('title', '')} {row.get('abstract', '')}".strip()


def _as_2d(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def rank_papers(
    question: str,
    papers: pd.DataFrame,
    embedder: Embedder | None = None,
) -> pd.DataFrame:
    """Score papers by cosine similarity between the question and title+abstract."""
    ranked = papers.copy()
    if ranked.empty:
        ranked["similarity"] = []
        return ranked

    model = embedder or get_embedder()
    question_embedding = _as_2d(model.encode(question))
    texts = ranked.apply(paper_text, axis=1).tolist()
    paper_embeddings = _as_2d(model.encode(texts))
    similarities = cosine_similarity(question_embedding, paper_embeddings)[0]
    ranked["similarity"] = similarities
    return ranked.sort_values("similarity", ascending=False).reset_index(drop=True)


def filter_relevant(papers: pd.DataFrame, min_similarity: float = 0.35) -> pd.DataFrame:
    """Drop papers that are not semantically close to the question."""
    if papers.empty or "similarity" not in papers.columns:
        return papers.copy()
    return papers.loc[papers["similarity"] >= min_similarity].reset_index(drop=True)
