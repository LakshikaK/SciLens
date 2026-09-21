import pandas as pd

from scilens.classifier import classify_papers
from scilens.consensus import aggregate_evidence
from scilens.pipeline import analyze_question


def fake_classifier_factory(mapping):
    def _classify(question, abstract):
        for needle, result in mapping.items():
            if needle.lower() in abstract.lower():
                return result
        return {"label": "Mixed", "score": 0.4, "raw_label": "Mixed"}

    return _classify


def test_aggregate_supports_when_majority_agree():
    papers = pd.DataFrame(
        {
            "label": ["Supports", "Supports", "Mixed"],
            "confidence": [0.9, 0.8, 0.5],
            "similarity": [0.7, 0.6, 0.5],
        }
    )
    result = aggregate_evidence(papers)
    assert result["verdict"] == "Supports"
    assert result["counts"]["Supports"] == 2
    assert 0 < result["confidence"] < 1


def test_aggregate_insufficient_when_empty():
    result = aggregate_evidence(pd.DataFrame())
    assert result["verdict"] == "Insufficient evidence"
    assert result["confidence"] == 0.0


def test_pipeline_with_injected_papers_and_classifier():
    papers = pd.DataFrame(
        {
            "pmid": ["1", "2", "3"],
            "title": [
                "Creatine and cognition",
                "Creatine memory trial",
                "Unrelated muscle gene therapy",
            ],
            "abstract": [
                "Creatine improved cognition and memory in adults in this trial. " * 2,
                "Creatine did not improve cognition compared with placebo. " * 2,
                "Gene therapy restored dystrophin in muscle without cognitive tests. " * 2,
            ],
            "year": ["2024", "2023", "2022"],
            "journal": ["A", "B", "C"],
            "url": ["u1", "u2", "u3"],
        }
    )

    class Embedder:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for text in texts:
                t = text.lower()
                vectors.append(
                    [
                        1.0 if "creatine" in t else 0.0,
                        1.0 if "cognition" in t or "memory" in t else 0.0,
                        1.0 if "gene" in t or "dystrophin" in t else 0.0,
                    ]
                )
            return vectors

    result = analyze_question(
        "Does creatine improve cognition?",
        min_similarity=0.4,
        papers=papers,
        embedder=Embedder(),
        classifier=fake_classifier_factory(
            {
                "improved cognition": {"label": "Supports", "score": 0.9, "raw_label": "Supports"},
                "did not improve": {"label": "Contradicts", "score": 0.85, "raw_label": "Contradicts"},
            }
        ),
    )
    assert result.n_retrieved == 3
    assert result.n_relevant == 2
    assert result.counts["Supports"] == 1
    assert result.counts["Contradicts"] == 1
    assert result.verdict in {"Mixed evidence", "Supports", "Contradicts"}
    discarded_titles = " ".join(row["title"] for row in result.discarded)
    assert "gene therapy" in discarded_titles.lower()


def test_classify_papers_uses_injected_fn():
    papers = pd.DataFrame(
        {
            "abstract": [
                "Creatine improved working memory performance in healthy adults and executive function overall."
            ]
        }
    )
    out = classify_papers(
        "Creatine improves cognition",
        papers,
        classifier=lambda q, a: {"label": "supports: x", "score": 0.77, "raw_label": "supports: x"},
    )
    assert out.iloc[0]["label"] == "Supports"
    assert out.iloc[0]["confidence"] == 0.77
