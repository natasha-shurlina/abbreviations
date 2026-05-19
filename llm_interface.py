from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def ask(self, fragment: str, context: str) -> str:
        pass