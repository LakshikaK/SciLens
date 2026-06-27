from classifier import classify_paper

question = "Creatine improves cognition"

abstract = """
Creatine supplementation improved memory
performance and executive function in healthy adults.
"""

print(
    classify_paper(
        question,
        abstract
    )
)