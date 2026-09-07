"""llama.cpp-style model metadata, read from a model directory's config.json."""

import json
import pathlib
from functools import lru_cache
from typing import Optional

from endpoints.core.types.model import ModelCardMeta


def read_model_meta(
    model_dir: pathlib.Path, n_ctx: Optional[int] = None, include_size: bool = False
) -> Optional[ModelCardMeta]:
    """
    Build the ``meta`` block clients expect from llama-server: the trained context
    length, vocabulary and embedding sizes from config.json, optionally the loaded
    context length and the weight file size. Returns None when the directory has no
    readable config.json. Results are cached per directory and config.json mtime,
    since model listings are polled frequently.
    """

    config_path = model_dir / "config.json"
    try:
        mtime = config_path.stat().st_mtime_ns
    except OSError:
        return None

    meta = _read_config_meta(config_path, mtime)
    if meta is None:
        return None

    meta = meta.model_copy()
    if n_ctx is not None:
        meta.n_ctx = n_ctx
    if include_size:
        meta.size = _weights_size(model_dir, mtime)

    return meta


@lru_cache(maxsize=256)
def _read_config_meta(config_path: pathlib.Path, mtime: int) -> Optional[ModelCardMeta]:
    try:
        cfg = json.loads(config_path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None

    # Multimodal configs nest the text model's settings under text_config
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    merged = {**cfg, **text_cfg}

    return ModelCardMeta(
        n_ctx_train=int(merged.get("max_position_embeddings") or 0),
        n_vocab=int(merged.get("vocab_size") or 0),
        n_embd=int(merged.get("hidden_size") or 0),
    )


@lru_cache(maxsize=64)
def _weights_size(model_dir: pathlib.Path, mtime: int) -> int:
    try:
        return sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    except OSError:
        return 0
