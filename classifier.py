from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_paper(question, abstract):

    labels = [
        f"supports: {question}",
        f"contradicts: {question}",
        f"mixed evidence regarding: {question}"
    ]

    result = classifier(
        abstract,
        labels
    )

    return {
        "label": result["labels"][0],
        "score": result["scores"][0]
    }