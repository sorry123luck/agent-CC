#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OmniParser standalone worker — launched by WorkerManager.

Replaces the inline 500-char Python script that was previously constructed
inside ``OmniParserRemoteVisionProvider._start_local_server()``.

Protocol:
  - stdout: JSON-lines events  {"event": "ready", "port": 8001}
  - stderr: forwarded to log file
  - Shutdown: SIGTERM / SIGINT → graceful exit

Usage (called by WorkerManager, not by humans):
  python scripts/omniparser_worker.py \
    --project-root D:\\...\\vendor\\omniparser_runtime \
    --som-model-path D:\\...\\model.pt \
    --caption-model-name florence2 \
    --caption-model-path D:\\...\\icon_caption_florence \
    --host 127.0.0.1 \
    --port 8001 \
    --cache-dir D:\\...\\.cache
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import types
import importlib
import importlib.machinery


def _emit(payload: dict) -> None:
    """Write a JSON-line to stdout."""
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _install_flash_attn_stub() -> None:
    """Stub out flash_attn so Florence-2 loads without the real package."""
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)

    def _unsupported(*args, **kwargs):
        raise RuntimeError("flash_attn is unavailable on this local OmniParser runtime")

    bert_padding.index_first_axis = _unsupported
    bert_padding.pad_input = _unsupported
    bert_padding.unpad_input = _unsupported
    flash_attn.bert_padding = bert_padding
    sys.modules.setdefault("flash_attn", flash_attn)
    sys.modules.setdefault("flash_attn.bert_padding", bert_padding)


def _patch_transformers() -> None:
    """Patch transformers to skip flash_attn imports and use local models."""
    import transformers.utils as tr_utils
    from transformers.utils import import_utils as tr_import_utils
    import transformers.dynamic_module_utils as tr_dynamic_utils

    tr_utils.is_flash_attn_2_available = lambda: False
    tr_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    tr_import_utils.is_flash_attn_2_available = lambda: False
    tr_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

    _original_check_imports = tr_dynamic_utils.check_imports

    def _patched_check_imports(filename):
        imports = [imp for imp in tr_dynamic_utils.get_imports(filename) if imp != "flash_attn"]
        missing = []
        for imp in imports:
            try:
                importlib.import_module(imp)
            except ImportError:
                missing.append(imp)
        if missing:
            raise ImportError(
                "This modeling file requires the following packages that were not found in your environment: "
                + ", ".join(missing)
                + ". Run `pip install "
                + " ".join(missing)
                + "`"
            )
        return tr_dynamic_utils.get_relative_imports(filename)

    tr_dynamic_utils.check_imports = _patched_check_imports


def _patch_caption_model() -> None:
    """Patch OmniParser caption model loader for local Florence-2."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    import util.utils as omn_utils

    _original = omn_utils.get_caption_model_processor

    def _patched(model_name, model_name_or_path, device=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name != "florence2":
            return _original(model_name, model_name_or_path, device=device)
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)
        torch_dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        ).to(device)
        return {"model": model.to(device), "processor": processor}

    omn_utils._original_get_caption_model_processor = _original
    omn_utils.get_caption_model_processor = _patched


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniParser standalone worker")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--som-model-path", required=True)
    parser.add_argument("--caption-model-name", default="florence2")
    parser.add_argument("--caption-model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args()

    # Set cache dirs
    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Add project paths
    sys.path.insert(0, args.project_root)
    sys.path.insert(0, os.path.join(args.project_root, "omnitool"))

    # Install patches
    _install_flash_attn_stub()
    _patch_transformers()
    _patch_caption_model()

    # Set sys.argv for uvicorn/omniparser
    sys.argv = [
        "omniparserserver",
        "--som_model_path", args.som_model_path,
        "--caption_model_name", args.caption_model_name,
        "--caption_model_path", args.caption_model_path,
        "--host", args.host,
        "--port", str(args.port),
    ]

    # Signal handler for graceful shutdown
    def _shutdown(signum, frame):
        _emit({"event": "shutting_down", "signal": signum})
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Emit ready event (port may differ from requested if 0)
    _emit({"event": "ready", "port": args.port, "host": args.host})

    # Start server
    import uvicorn
    from omniparserserver.omniparserserver import app

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
