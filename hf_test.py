from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = """
Creatine supplementation improved memory
performance and executive function in healthy adults.
"""

labels = [
    "supports claim",
    "contradicts claim",
    "mixed evidence"
]

result = classifier(
    text,
    labels
)

print(result)
