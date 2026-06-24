from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def classify_paper(question, abstract):

    prompt = f"""
Question:
{question}

Abstract:
{abstract}

Your task:

Determine whether the paper:

- Supports the claim
- Contradicts the claim
- Provides mixed evidence

Return ONLY one word:

Supports
Contradicts
Mixed
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()