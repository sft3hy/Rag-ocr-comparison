import os
from groq import Groq


class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def create_chat_completion(
        self, model: str, messages: list, temperature: float, max_tokens: int
    ):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
