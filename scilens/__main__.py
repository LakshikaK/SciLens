from __future__ import annotations

import argparse
import json
import sys

from scilens.pipeline import analyze_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the scientific evidence around a research question."
    )
    parser.add_argument("question", help='Claim or question, e.g. "Does creatine improve cognition?"')
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument(
        "--backend",
        choices=["nli", "openai"],
        default="nli",
        help="nli uses a local transformer; openai needs OPENAI_API_KEY",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)

    result = analyze_question(
        args.question,
        max_results=args.max_results,
        min_similarity=args.min_similarity,
        backend=args.backend,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\nQuestion: {result.question}")
    print(f"Papers retrieved: {result.n_retrieved}")
    print(f"Relevant after ranking/filter: {result.n_relevant}")
    print(f"\nConsensus: {result.verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\n{result.explanation}\n")
    print("Evidence mix")
    print(f"  Supports:     {result.counts['Supports']}")
    print(f"  Contradicts:  {result.counts['Contradicts']}")
    print(f"  Mixed:        {result.counts['Mixed']}")
    if result.n_relevant:
        print("\nTop relevant papers")
        for paper in result.papers[:8]:
            print(
                f"  [{paper['label']}] {paper['similarity']:.2f}  "
                f"{paper['title'][:90]}  PMID {paper['pmid']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
