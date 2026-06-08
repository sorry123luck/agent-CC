"""Unified path management for Vision Runtime.

All paths resolve from config + environment variables, never hardcoded.

Key distinction:
- ``repo_root``: the git repository root (where scripts/, src/, vendor/ live)
- ``omniparser_runtime``: vendor/omniparser_runtime (OmniParser's own code)
- ``data_dir`` / ``logs_dir`` / ``temp_dir``: always under repo_root/data/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class RuntimePaths:
    """Centralized path descriptors for all Vision Runtime components.

    Built via ``from_config()`` which reads config/models.yaml values
    and environment variables, with sensible project-relative defaults.

    Field semantics:
    - ``repo_root``: git repository root for this project
    - ``omniparser_runtime``: vendor/omniparser_runtime directory
    - ``worker_script``: scripts/omniparser_worker.py (always under repo_root)
    - ``omniparser_weights``: models/omniparser/weights (the weights root)
    - ``som_model_path``: direct path to icon_detect/model.pt (from config)
    - ``caption_model_path``: direct path to icon_caption_florence (from config)
    - ``cache_dir``: HuggingFace cache dir (from config)
    """

    repo_root: Path
    data_dir: Path
    logs_dir: Path
    temp_dir: Path
    ocr_python: Path
    ocr_persistent_script: Path
    worker_script: Path
    omniparser_runtime: Path
    omniparser_weights: Path
    omniparser_log_dir: Path
    som_model_path: Path
    caption_model_path: Path
    cache_dir: Path

    @classmethod
    def from_config(
        cls,
        vision_cfg: dict[str, Any] | None = None,
        ocr_cfg: dict[str, Any] | None = None,
    ) -> RuntimePaths:
        """Build from config dicts (typically from config/models.yaml).

        Falls back to project-relative defaults when config keys are absent.
        Environment variables override config for sensitive paths.
        """
        vc = vision_cfg or {}
        oc = ocr_cfg or {}

        # repo_root is always the git repo, NOT the OmniParser runtime
        repo_root = _REPO_DIR

        # OCR python: env > config > default
        ocr_python = Path(
            os.environ.get("OPENCLAW_OCR_PYTHON")
            or oc.get("worker_python")
            or r"D:\ocr-paddle-env\Scripts\python.exe"
        )

        # OmniParser runtime root (vendor/omniparser_runtime)
        omniparser_runtime = Path(
            vc.get("project_root") or str(repo_root / "vendor" / "omniparser_runtime")
        )

        # Weights root: derive correctly from som_model_path
        som_model_path = Path(
            vc.get("som_model_path")
            or str(repo_root / "models" / "omniparser" / "weights" / "icon_detect" / "model.pt")
        )
        caption_model_path = Path(
            vc.get("caption_model_path")
            or str(repo_root / "models" / "omniparser" / "weights" / "icon_caption_florence")
        )
        cache_dir = Path(
            vc.get("cache_dir")
            or str(repo_root / "models" / "omniparser" / "weights" / ".cache")
        )

        # Weights root: walk up from som_model_path
        # som_model_path = .../weights/icon_detect/model.pt
        # weights root = .../weights
        if som_model_path.name == "model.pt" and som_model_path.parent.name == "icon_detect":
            omniparser_weights = som_model_path.parent.parent
        else:
            omniparser_weights = repo_root / "models" / "omniparser" / "weights"

        return cls(
            repo_root=repo_root,
            data_dir=repo_root / "data",
            logs_dir=repo_root / "data" / "inspect" / "runtime",
            temp_dir=repo_root / "data" / "temp",
            ocr_python=ocr_python,
            ocr_persistent_script=Path(
                oc.get("persistent_worker_script")
                or str(repo_root / "scripts" / "paddleocr_persistent_worker.py")
            ),
            worker_script=repo_root / "scripts" / "omniparser_worker.py",
            omniparser_runtime=omniparser_runtime,
            omniparser_weights=omniparser_weights,
            omniparser_log_dir=repo_root / "data" / "inspect" / "runtime",
            som_model_path=som_model_path,
            caption_model_path=caption_model_path,
            cache_dir=cache_dir,
        )
