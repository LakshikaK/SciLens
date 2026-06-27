import pandas as pd
from classifier import classify_paper

question = "Creatine improves cognition"

papers = pd.read_csv("data/papers.csv")

labels = []
scores = []

for i, row in papers.iterrows():

    print(f"Processing paper {i+1}/{len(papers)}")

    result = classify_paper(
        question,
        str(row["Abstract"])
    )

    labels.append(result["label"])
    scores.append(result["score"])

papers["Classification"] = labels
papers["Confidence"] = scores

print(papers[["Title", "Classification", "Confidence"]].head())