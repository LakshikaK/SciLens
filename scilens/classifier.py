from __future__ import annotations

import os
from typing import Any, Callable

import pandas as pd

LABELS = ("Supports", "Contradicts", "Mixed")
DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"

_nli_pipeline: Any = None


ClassifierFn = Callable[[str, str], dict[str, Any]]


def normalize_label(raw: str) -> str:
    text = (raw or "").lower()
    if "contradict" in text:
        return "Contradicts"
    if "mixed" in text or "inconclusive" in text or "unclear" in text:
        return "Mixed"
    if "support" in text:
        return "Supports"
    return "Mixed"


def get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline

        model_name = os.getenv("SCILENS_NLI_MODEL", DEFAULT_NLI_MODEL)
        _nli_pipeline = pipeline("zero-shot-classification", model=model_name)
    return _nli_pipeline


def classify_with_nli(
    question: str,
    abstract: str,
    nli=None,
) -> dict[str, Any]:
    """Zero-shot NLI: does this abstract support, contradict, or mix the claim?"""
    claim = question.rstrip("?").strip()
    labels = [
        f"supports: {claim}",
        f"contradicts: {claim}",
        f"mixed evidence regarding: {claim}",
    ]
    clf = nli or get_nli_pipeline()
    result = clf(abstract, labels)
    return {
        "label": normalize_label(result["labels"][0]),
        "score": float(result["scores"][0]),
        "raw_label": result["labels"][0],
    }


def classify_with_openai(question: str, abstract: str, client=None) -> dict[str, Any]:
    """Optional LLM backend. Requires OPENAI_API_KEY."""
    from openai import OpenAI

    api_client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Question/claim:
{question}

Abstract:
{abstract}

Decide whether the paper:
- Supports the claim
- Contradicts the claim
- Provides mixed evidence

Return ONLY one word: Supports, Contradicts, or Mixed."""
    response = api_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    return {
        "label": normalize_label(content),
        "score": 0.7,
        "raw_label": content,
    }


def get_classifier(backend: str = "nli") -> ClassifierFn:
    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for the openai backend.")
        return classify_with_openai
    if backend == "nli":
        return classify_with_nli
    raise ValueError(f"Unknown classifier backend: {backend}")


def classify_papers(
    question: str,
    papers: pd.DataFrame,
    backend: str = "nli",
    classifier: ClassifierFn | None = None,
) -> pd.DataFrame:
    classified = papers.copy()
    if classified.empty:
        classified["label"] = []
        classified["confidence"] = []
        classified["raw_label"] = []
        return classified

    fn = classifier or get_classifier(backend)
    labels: list[str] = []
    scores: list[float] = []
    raw_labels: list[str] = []

    for _, row in classified.iterrows():
        abstract = str(row.get("abstract") or "")
        if len(abstract) < 40:
            labels.append("Mixed")
            scores.append(0.0)
            raw_labels.append("skipped-short-abstract")
            continue
        try:
            result = fn(question, abstract)
            labels.append(normalize_label(str(result["label"])))
            scores.append(float(result.get("score") or 0.0))
            raw_labels.append(str(result.get("raw_label") or result["label"]))
        except Exception as exc:  # classification should not abort the whole run
            labels.append("Mixed")
            scores.append(0.0)
            raw_labels.append(f"error:{exc}")

    classified["label"] = labels
    classified["confidence"] = scores
    classified["raw_label"] = raw_labels
    return classified
