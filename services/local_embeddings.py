"""Qwen3-VL multimodal embeddings for project vector collections."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import huggingface_hub.file_download

huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False

PROJECT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
PROJECT_EMBEDDING_DIMENSION = 2048
PROJECT_EMBEDDING_DEVICE = os.getenv("PROJECT_EMBEDDING_DEVICE", "cpu")
PROJECT_EMBEDDING_MAX_LENGTH = 8192
PROJECT_IMAGE_MIN_PIXELS = 4 * 32 * 32
PROJECT_IMAGE_MAX_PIXELS = 1800 * 32 * 32
DOCUMENT_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
_MODEL_LOCK = threading.RLock()
_PROJECT_EMBEDDER: Any = None


def get_project_embedding_function() -> Any:
    global _PROJECT_EMBEDDER
    with _MODEL_LOCK:
        if _PROJECT_EMBEDDER is not None:
            return _PROJECT_EMBEDDER

        _PROJECT_EMBEDDER = _load_project_embedding_function()
        return _PROJECT_EMBEDDER


def _load_project_embedding_function() -> Any:
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    if PROJECT_EMBEDDING_DEVICE == "cpu":
        torch.set_num_threads(min(os.cpu_count() or 1, 4))

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def forward(self, **model_inputs: Any) -> Any:
            return self.model(**model_inputs)

    class Qwen3VLEmbedder:
        def __init__(self) -> None:
            self.device = torch.device(PROJECT_EMBEDDING_DEVICE)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "Qwen3-VL embeddings require a CUDA-enabled PyTorch "
                    "installation by default, but CUDA is unavailable. Install "
                    "a CUDA-enabled PyTorch build or explicitly set "
                    "PROJECT_EMBEDDING_DEVICE=cpu to opt into slow CPU inference."
                )
            self.model = Qwen3VLForEmbedding.from_pretrained(
                PROJECT_EMBEDDING_MODEL,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            self.processor = Qwen3VLProcessor.from_pretrained(
                PROJECT_EMBEDDING_MODEL,
                padding_side="right",
            )

        @torch.inference_mode()
        def process(
            self,
            records: list[dict[str, Any]],
            *,
            normalize: bool = True,
        ) -> Any:
            from qwen_vl_utils.vision_process import process_vision_info

            conversations = []
            for record in records:
                content = []
                image = record.get("image")
                if image is not None:
                    content.append({
                        "type": "image",
                        "image": os.fspath(image),
                        "min_pixels": PROJECT_IMAGE_MIN_PIXELS,
                        "max_pixels": PROJECT_IMAGE_MAX_PIXELS,
                    })
                text = record.get("text")
                if text:
                    content.append({"type": "text", "text": text})
                if not content:
                    raise ValueError("Each embedding input needs text or an image.")
                conversations.append([
                    {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": record.get(
                                "instruction",
                                DOCUMENT_INSTRUCTION,
                            ),
                        }],
                    },
                    {"role": "user", "content": content},
                ])

            prompt_text = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=False,
            )
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            if video_inputs is not None:
                videos, video_metadata = zip(*video_inputs)
                videos = list(videos)
                video_metadata = list(video_metadata)
            else:
                videos, video_metadata = None, None

            model_inputs = self.processor(
                text=prompt_text,
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                truncation=True,
                max_length=PROJECT_EMBEDDING_MAX_LENGTH,
                padding=True,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )
            model_inputs = {
                name: value.to(self.device)
                for name, value in model_inputs.items()
            }
            outputs = self.model(**model_inputs)
            attention_mask = model_inputs["attention_mask"]
            last_token = (
                attention_mask.shape[1]
                - attention_mask.flip(dims=[1]).argmax(dim=1)
                - 1
            )
            row = torch.arange(outputs.last_hidden_state.shape[0], device=self.device)
            embeddings = outputs.last_hidden_state[row, last_token].to(
                dtype=torch.float32,
            )
            if normalize:
                embeddings = torch.nn.functional.normalize(
                    embeddings,
                    p=2,
                    dim=-1,
                )
            return embeddings.cpu().numpy()

    return Qwen3VLEmbedder()


@lru_cache(maxsize=1)
def get_project_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(PROJECT_EMBEDDING_MODEL)


def count_embedding_tokens(text: str) -> int:
    """Count special tokens using the Qwen model tokenizer."""
    return len(get_project_tokenizer().encode(text, add_special_tokens=True))


def embed_records(
    records: list[dict[str, Any]],
    *,
    model_loaded_callback: Callable[[], None] | None = None,
) -> Any:
    """Embed text, images, or combined text-and-image records with Qwen."""
    if not records:
        return []
    with _MODEL_LOCK:
        embedder = get_project_embedding_function()
        if model_loaded_callback is not None:
            model_loaded_callback()
        embeddings = embedder.process(
            records,
            normalize=True,
        )
    return _validate_embeddings(embeddings, len(records))


def embed_texts(texts: list[str]) -> Any:
    """Embed document text in Qwen's shared multimodal vector space."""
    return embed_records([
        {"text": text, "instruction": DOCUMENT_INSTRUCTION}
        for text in texts
    ])


def embed_text_query(text: str) -> Any:
    """Embed a search question using Qwen's query instruction."""
    return embed_records([{
        "text": text,
        "instruction": QUERY_INSTRUCTION,
    }])[0]


def embed_images(image_paths: list[str | Path]) -> Any:
    """Embed image files directly in Qwen's shared vector space."""
    return embed_records([
        {"image": str(path), "instruction": DOCUMENT_INSTRUCTION}
        for path in image_paths
    ])


def _validate_embeddings(embeddings: Any, expected_count: int) -> Any:
    import numpy as np

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_count:
        raise ValueError(
            "Qwen must return exactly one embedding vector per input record."
        )
    if vectors.shape[1] != PROJECT_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Qwen embedding dimension mismatch: expected "
            f"{PROJECT_EMBEDDING_DIMENSION}, got {vectors.shape[1]}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Qwen embeddings contain non-finite values.")
    return vectors
