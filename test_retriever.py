import pandas as pd
from retriever import rank_papers

papers = pd.read_csv("data/papers.csv")

ranked = rank_papers(
    "Creatine improves cognition",
    papers
)

print(
    ranked[
        ["Title", "Similarity"]
    ].head(10)
)