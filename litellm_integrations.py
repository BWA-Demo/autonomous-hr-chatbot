"""LangChain 0.0.x adapters that call OpenAI-compatible models through LiteLLM."""

from __future__ import annotations

import os
from typing import Any, List, Optional

import litellm
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.embeddings.base import Embeddings
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Ensure LiteLLM is used for completions/embeddings (explicit package use).
__all__ = ["LitellmChatModel", "LitellmEmbeddings"]


def _messages_to_litellm(messages: List[BaseMessage]) -> List[dict]:
    out: List[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        else:
            out.append({"role": "user", "content": getattr(m, "content", str(m))})
    return out


class LitellmEmbeddings(Embeddings):
    """Embeddings via ``litellm.embedding`` (OpenAI and other providers supported by LiteLLM)."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.api_version = api_version

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        kwargs: dict = {"model": self.model, "input": texts, "api_key": self.api_key}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_version:
            kwargs["api_version"] = self.api_version
        resp = litellm.embedding(**kwargs)
        data = resp.get("data") or []
        return [row["embedding"] for row in sorted(data, key=lambda x: x.get("index", 0))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LitellmChatModel(SimpleChatModel):
    """Chat model for LangChain agents/chains, backed by ``litellm.completion``."""

    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    openai_api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "litellm"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **_kwargs: Any,
    ) -> str:
        api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        req: dict[str, Any] = {
            "model": self.model_name,
            "messages": _messages_to_litellm(messages),
            "temperature": self.temperature,
            "stop": stop,
            "api_key": api_key,
        }
        if self.api_base:
            req["api_base"] = self.api_base
        if self.api_version:
            req["api_version"] = self.api_version
        resp = litellm.completion(**req)
        return resp.choices[0].message.content
