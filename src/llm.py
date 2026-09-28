import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    ROOT_DIR / ".env",
    override=True
)


class LLMClient:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing."
            )

        if not base_url:
            raise ValueError(
                "OPENAI_BASE_URL is missing."
            )

        if not model_name:
            raise ValueError(
                "OPENAI_MODEL is missing."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model_name = model_name

    def answer(
        self,
        query,
        results
    ):

        context_parts = []

        for i, r in enumerate(results, 1):

            context_parts.append(
                f"""
Source [{i}]
Paper: {r['source']}
Page: {r['page']}

Content:
{r['text']}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are an expert in waste thermochemical conversion.

Answer the question only based on the provided context.

If the context is insufficient,
clearly say so.

Use citations such as [1], [2].

Context:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content