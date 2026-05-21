from abc import ABC, abstractmethod
from typing import List, Tuple

class LLMProvider(ABC):
    @abstractmethod
    def ask(self, fragment: str, context: str) -> str:
        pass

    def ask_batch(self, fragments_contexts: List[Tuple[str, str]]) -> List[str]:
        return [self.ask(frag, ctx) for frag, ctx in fragments_contexts]