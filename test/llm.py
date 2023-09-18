from hashlib import md5
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings import FakeEmbeddings
from langchain.llms.fake import FakeListLLM

from coml.constants import EMBED_DIM


class MockKnowledgeLLM(FakeListLLM):
    responses: List[str] = ["This is a mock knowledge."]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = self.responses[0]
        return response


class MockSuggestLLM(FakeListLLM):
    responses: List[str] = [
        (
            "Configuration 1: cost is very small. gamma is very small. kernel is linear. degree is very small.\n"
            "Configuration 2: cost is very small. gamma is very small. kernel is linear. degree is very small.\n"
            "Configuration 3: cost is very small. gamma is very small. kernel is linear. degree is very small."
        )
    ]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = self.responses[0]
        return response


class MockEmbeddingModel(FakeEmbeddings):
    size: int = 1536

    def _get_embedding(self, text) -> List[float]:
        md5_10 = int(md5(text.encode("utf-8")).hexdigest(), 16)
        return [md5_10 // 10**i % 10 for i in range(10)] + [0.0] * (EMBED_DIM - 10)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)
