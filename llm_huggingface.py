import os
from huggingface_hub import InferenceClient
from llm_interface import LLMProvider

class HuggingFaceProvider(LLMProvider):
    def __init__(self, api_token: str = None, model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        if not self.api_token:
            raise ValueError("HF_TOKEN не задан")
        self.model = model
        self.client = InferenceClient(token=self.api_token, model=self.model)

    def ask(self, fragment: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "Ты помощник по русскому языку и инженерной документации. Отвечай коротко одним словом или словосочетанием без пояснений."},
            {"role": "user", "content": f"В тексте инженерного документа встретилось сокращение «{fragment}». Контекст: {context}. Напиши полное слово, которое было сокращено. Если это не сокращение, ответь: Не сокращение."}
        ]
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                max_tokens=64,
                temperature=0.0
            )
            ans = response.choices[0].message.content.strip().lower()
            if ans == "Не сокращение":
                return "Не сокращение"
            else:
                return f"неправильное сокращение, возможно сокращено слово {ans}"
        except Exception as e:
            return f"(ошибка LLM: {e})"