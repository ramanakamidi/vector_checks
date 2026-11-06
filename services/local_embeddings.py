"""Torch-backed local embeddings for project Chroma collections."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import huggingface_hub.file_download

huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False


@lru_cache(maxsize=1)
def get_project_embedding_function() -> Any:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True,
    )


@lru_cache(maxsize=1)
def get_project_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def count_embedding_tokens(text: str) -> int:
    """Count special tokens too, matching the embedding model's input budget."""
    return len(get_project_tokenizer().encode(text, add_special_tokens=True))


def embed_texts(texts: list[str]) -> Any:
    """Embed texts explicitly so collections keep their persisted config."""
    return get_project_embedding_function()(texts)