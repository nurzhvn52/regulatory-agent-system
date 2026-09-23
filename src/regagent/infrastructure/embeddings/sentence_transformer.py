"""Lazy Sentence Transformers adapter with normalized dense embeddings."""

import asyncio
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any, Protocol, cast

from regagent.application.retrieval.models import EmbeddingModelSpec


class _ArrayLike(Protocol):
    def tolist(self) -> list[Any]: ...


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> _ArrayLike: ...

    def get_embedding_dimension(self) -> int | None: ...


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        revision: str,
        dimensions: int,
        *,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        if dimensions < 1 or batch_size < 1:
            raise ValueError("dimensions and batch_size must be positive")
        self._model_name = model_name
        self._revision = revision
        self._dimensions = dimensions
        self._device = device
        self._batch_size = batch_size
        self._model: _SentenceTransformerModel | None = None
        self._load_lock = Lock()

    @property
    def model_spec(self) -> EmbeddingModelSpec:
        return EmbeddingModelSpec(
            name=self._model_name,
            revision=self._revision,
            normalized=True,
            config={
                "backend": "sentence-transformers",
                "backend_version": _package_version(),
                "device": self._device,
                "normalize_embeddings": True,
            },
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_many, tuple(texts))

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    def _encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._get_model()
        encoded = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        raw = encoded.tolist()
        vectors = cast(list[list[float]], raw)
        if any(len(vector) != self._dimensions for vector in vectors):
            raise ValueError(
                f"Expected {self._dimensions}-dimensional embeddings from {self._model_name}"
            )
        return vectors

    def _get_model(self) -> _SentenceTransformerModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                try:
                    module = import_module("sentence_transformers")
                except ImportError as error:
                    raise RuntimeError(
                        "sentence-transformers is not installed; run 'poetry install'"
                    ) from error
                model_class = getattr(module, "SentenceTransformer")  # noqa: B009
                device = None if self._device == "auto" else self._device
                loaded = cast(
                    _SentenceTransformerModel,
                    model_class(
                        self._model_name,
                        revision=self._revision,
                        device=device,
                    ),
                )
                actual_dimensions = loaded.get_embedding_dimension()
                if actual_dimensions != self._dimensions:
                    raise ValueError(
                        f"Configured dimensions {self._dimensions} do not match model "
                        f"dimensions {actual_dimensions}"
                    )
                self._model = loaded
        return self._model


def _package_version() -> str:
    try:
        return version("sentence-transformers")
    except PackageNotFoundError:
        return "not-installed"
