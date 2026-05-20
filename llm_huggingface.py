import os
from typing import List, Tuple
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
        return self.ask_batch([(fragment, context)])[0]

    def ask_batch(self, fragments_contexts: List[Tuple[str, str]]) -> List[str]:
        if not fragments_contexts:
            return []
        prompt_lines = []
        for i, (frag, ctx) in enumerate(fragments_contexts, 1):
            prompt_lines.append(f"{i}. Сокращение «{frag}» в контексте: «{ctx}»")
        prompt = (
            "Ты помощник по русскому языку и инженерной документации.\n"
            "Расшифруй каждое из следующих сокращений. Ответь для каждого пункта полным словом "
            "или фразой «не сокращение». Перечисли ответы в том же порядке, разделяя символом |.\n\n"
            + "\n".join(prompt_lines)
        )
        messages = [
            {"role": "system", "content": "Ты полезный ассистент."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                max_tokens=256,
                temperature=0.0
            )
            ans = response.choices[0].message.content.strip()
            parts = [p.strip() for p in ans.split('|')]
            while len(parts) < len(fragments_contexts):
                parts.append("")
            processed = []
            for p in parts[:len(fragments_contexts)]:
                p_lower = p.lower()
                if p_lower == "не сокращение":
                    processed.append("не сокращение")
                else:
                    processed.append(f"Возможно сокращено слово {p}")
            return processed
        except Exception as e:
            return [f"(ошибка LLM: {e})"] * len(fragments_contexts)