from llm import classify_paper

question = "Does creatine improve cognition?"

abstract = """
Creatine supplementation improved memory
performance and executive function in healthy
adults compared to placebo.
"""

result = classify_paper(
    question,
    abstract
)

print(result)