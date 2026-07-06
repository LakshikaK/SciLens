import pandas as pd
from classifier import classify_paper

question = "Creatine improves cognition"

papers = pd.read_csv("data/papers.csv")

labels = []
scores = []

for i, row in papers.iterrows():

    print(f"Processing paper {i+1}/{len(papers)}")

    try:
        result = classify_paper(
            question,
            str(row["Abstract"])
        )

        labels.append(result["label"])
        scores.append(result["score"])

    except Exception as e:
        print(f"Error on paper {i+1}: {e}")

        labels.append("Error")
        scores.append(None)
        
papers["Classification"] = labels
papers["Confidence"] = scores

print(papers[["Title", "Classification", "Confidence"]].head())

papers.to_csv(
    "data/classified_papers.csv",
    index=False
)

print("\nSaved classified_papers.csv")
