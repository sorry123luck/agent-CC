"""Remote vision provider and local OmniParser adapter."""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.common.utils import read_yaml
from src.perception.page_compiler_models import InteractionCanvas, SurfaceType, WindowInfoSnapshot
from src.perception.providers.base import PerceptionProviderBase

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OMNI_RUNTIME = _PROJECT_ROOT / "vendor" / "omniparser_runtime"
_DEFAULT_OMNI_WEIGHTS = _PROJECT_ROOT / "models" / "omniparser" / "weights"


@dataclass
class VisionCandidateElement:
    """Vision candidate returned by remote parser."""

    element_id: str
    bounding_box: tuple[int, int, int, int]
    semantic_label: str
    confidence: float
    icon_type: str | None = None
    text: str | None = None
    region_role: str | None = None
    group_id: str | None = None
    interaction_hints: dict[str, Any] | None = None
    structure_evidence_score: float | None = None
    attributes: dict[str, Any] | None = None


@dataclass
class VisionParseResult:
    """Normalized remote vision parse result."""

    candidates: list[VisionCandidateElement]
    visual_score: float | None = None
    layout_regions: list[dict[str, Any]] | None = None
    control_groups: list[dict[str, Any]] | None = None
    interaction_hints: list[dict[str, Any]] | None = None
    structure_evidence_score: float | None = None
    raw_output: dict[str, Any] | None = None
    provider: str = "omniparser"
    success: bool = False
    error: str | None = None


@dataclass
class RemoteVisionConfig:
    """Remote vision provider config."""

    provider: str = "omniparser"
    endpoint: str = "http://127.0.0.1:8001/parse/"
    probe_endpoint: str = "http://127.0.0.1:8001/probe/"
    timeout: int = 90
    retry: int = 2
    enabled: bool = True
    autostart: bool = True
    startup_timeout: int = 180
    python_path: str = sys.executable
    project_root: str = field(default_factory=lambda: str(_DEFAULT_OMNI_RUNTIME))
    caption_model_name: str = "florence2"
    som_model_path: str = field(
        default_factory=lambda: str(_DEFAULT_OMNI_WEIGHTS / "icon_detect" / "model.pt")
    )
    caption_model_path: str = field(
        default_factory=lambda: str(_DEFAULT_OMNI_WEIGHTS / "icon_caption_florence")
    )
    cache_dir: str = field(default_factory=lambda: str(_DEFAULT_OMNI_WEIGHTS / ".cache"))
    include_caption: bool = False
    include_ocr: bool = False


class IRemoteVisionProvider(PerceptionProviderBase):
    """Remote vision provider interface."""

    @property
    def name(self) -> str:
        return "RemoteVisionProvider"

    def supports(self, surface_type: SurfaceType) -> bool:
        return surface_type in (
            SurfaceType.NATIVE_UIA,
            SurfaceType.BROWSER,
            SurfaceType.ELECTRON_WEBVIEW,
            SurfaceType.CANVAS_SELF_DRAWN,
        )

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        raise NotImplementedError(f"{self.name}.extract() must be implemented")

    def parse_screenshot(self, image: Any) -> VisionParseResult:
        raise NotImplementedError(f"{self.name}.parse_screenshot() must be implemented")

    def detect_icon(self, image: Any, template_name: str) -> list[tuple[int, int]]:
        raise NotImplementedError(f"{self.name}.detect_icon() must be implemented")

    def get_visual_score(self, image: Any) -> float:
        result = self.parse_screenshot(image)
        return float(result.visual_score or 0.0)


class OmniParserRemoteVisionProvider(IRemoteVisionProvider):
    """Remote HTTP adapter for OmniParser-compatible endpoint."""

    def __init__(self, config_path: str | Path = "config/models.yaml") -> None:
        self._config_path = Path(config_path)
        self._config = self._load_config(self._config_path)
        self._server_process: subprocess.Popen[str] | None = None
        self._server_log_handle: Any | None = None
        self._raw_vision_cfg: dict[str, Any] = {}  # stored for WorkerManager paths

    @property
    def name(self) -> str:
        return "OmniParserRemoteVisionProvider"

    def parse_screenshot(self, image: Any) -> VisionParseResult:
        if not self._config.enabled:
            return VisionParseResult(
                candidates=[],
                provider=self._config.provider,
                success=False,
                error="vision_provider_disabled",
            )

        if not isinstance(image, Image.Image):
            return VisionParseResult(
                candidates=[],
                provider=self._config.provider,
                success=False,
                error="unsupported_image_type",
            )

        try:
            import requests
        except ImportError:
            return VisionParseResult(
                candidates=[],
                provider=self._config.provider,
                success=False,
                error="requests_not_installed",
            )

        if not self._ensure_service_available(requests):
            return VisionParseResult(
                candidates=[],
                provider=self._config.provider,
                success=False,
                error="omniparser_provider_unavailable",
            )

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        payload = None
        last_error: str | None = None

        for _ in range(max(1, self._config.retry)):
            try:
                response = requests.post(
                    self._config.endpoint,
                    json={
                        "base64_image": encoded_image,
                        "include_caption": self._config.include_caption,
                        "include_ocr": self._config.include_ocr,
                    },
                    timeout=self._config.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:
                last_error = str(exc)
                if self._config.autostart:
                    self._ensure_service_available(requests, force_restart=True)

        if payload is None:
            return VisionParseResult(
                candidates=[],
                provider=self._config.provider,
                success=False,
                error=last_error or "vision_provider_failed",
            )

        return self._normalize_payload(payload, image.size)

    def detect_icon(self, image: Any, template_name: str) -> list[tuple[int, int]]:
        result = self.parse_screenshot(image)
        positions: list[tuple[int, int]] = []
        for candidate in result.candidates:
            if candidate.icon_type == template_name and candidate.bounding_box:
                left, top, right, bottom = candidate.bounding_box
                positions.append(((left + right) // 2, (top + bottom) // 2))
        return positions

    def warm_up(self) -> dict[str, Any]:
        """Start the local OmniParser service before the first observe call."""
        status: dict[str, Any] = {
            "provider": self._config.provider,
            "ok": False,
            "include_caption": self._config.include_caption,
            "include_ocr": self._config.include_ocr,
        }
        if not self._config.enabled:
            status["error"] = "vision_provider_disabled"
            return status
        try:
            import requests
        except ImportError:
            status["error"] = "requests_not_installed"
            return status

        start = time.perf_counter()
        ok = self._ensure_service_available(requests)
        status["ok"] = ok
        status["elapsed_seconds"] = time.perf_counter() - start
        if not ok:
            status["error"] = "omniparser_provider_unavailable"
        return status

    def health_status(self) -> dict[str, Any]:
        """Return a non-mutating health probe for diagnostics endpoints."""
        status: dict[str, Any] = {
            "provider": self._config.provider,
            "enabled": self._config.enabled,
            "endpoint": self._config.endpoint,
            "probe_endpoint": self._config.probe_endpoint,
            "include_caption": self._config.include_caption,
            "include_ocr": self._config.include_ocr,
            "ok": False,
            "probe": {"ok": False},
        }
        if not self._config.enabled:
            status["probe"] = {"ok": False, "error": "vision_provider_disabled"}
            return status

        try:
            import requests
        except ImportError:
            status["probe"] = {"ok": False, "error": "requests_not_installed"}
            return status

        try:
            response = requests.get(
                self._config.probe_endpoint,
                timeout=min(5, self._config.timeout),
            )
            status["probe"] = {
                "ok": response.ok,
                "http_status": response.status_code,
            }
            if not response.ok:
                status["probe"]["error"] = f"vision_probe_http_{response.status_code}"
            status["ok"] = bool(response.ok)
        except Exception as exc:
            status["probe"] = {"ok": False, "error": str(exc)}
        return status

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        image_size: tuple[int, int],
    ) -> VisionParseResult:
        candidates_raw = (
            payload.get("parsed_elements")
            or payload.get("elements")
            or payload.get("candidates")
            or payload.get("parsed_content_list")
            or []
        )
        interaction_hints = payload.get("interaction_hints") or []
        interaction_hint_map = {
            str(item.get("candidate_id") or item.get("element_id") or item.get("id")): item
            for item in interaction_hints
            if isinstance(item, dict)
        }
        candidates: list[VisionCandidateElement] = []
        for index, item in enumerate(candidates_raw):
            bbox = self._extract_bbox(item, image_size=image_size)
            if bbox is None:
                continue
            candidate_id = str(item.get("element_id") or item.get("id") or f"vision_{index}")
            hint = interaction_hint_map.get(candidate_id, {})
            structure_evidence_score = self._coerce_float(
                item.get("structure_evidence_score", payload.get("structure_evidence_score"))
            )
            semantic_label = str(
                item.get("semantic_label")
                or item.get("type")
                or item.get("label")
                or "unknown"
            )
            text = item.get("text")
            if text is None:
                text = item.get("content")
            candidates.append(
                VisionCandidateElement(
                    element_id=candidate_id,
                    bounding_box=bbox,
                    semantic_label=semantic_label,
                    confidence=float(item.get("confidence", self._infer_confidence(item))),
                    icon_type=item.get("icon_type"),
                    text=text,
                    region_role=item.get("region_role"),
                    group_id=item.get("group_id"),
                    interaction_hints=hint if hint else None,
                    structure_evidence_score=structure_evidence_score,
                    attributes=dict(item),
                )
            )

        return VisionParseResult(
            candidates=candidates,
            visual_score=self._coerce_float(payload.get("visual_score")),
            layout_regions=payload.get("layout_regions"),
            control_groups=payload.get("control_groups"),
            interaction_hints=interaction_hints,
            structure_evidence_score=self._coerce_float(
                payload.get("structure_evidence_score", payload.get("structure_score"))
            ),
            raw_output=payload,
            provider=self._config.provider,
            success=True,
            error=None,
        )

    def _extract_bbox(
        self,
        item: dict[str, Any],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        bbox = item.get("bbox") or item.get("bounding_box") or item.get("box")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return self._normalize_bbox(bbox, image_size=image_size)

        coords = item.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            return self._normalize_bbox(coords, image_size=image_size)
        return None

    def _load_config(self, config_path: Path) -> RemoteVisionConfig:
        if not config_path.exists():
            return RemoteVisionConfig()
        raw = read_yaml(str(config_path)) or {}
        vision_cfg = (raw.get("models") or {}).get("vision") or {}
        endpoint = str(vision_cfg.get("endpoint", "http://127.0.0.1:8001/parse/"))
        probe_endpoint = str(vision_cfg.get("probe_endpoint") or self._derive_probe_endpoint(endpoint))
        # Store raw config for WorkerManager path resolution
        self._raw_vision_cfg = dict(vision_cfg)

        return RemoteVisionConfig(
            provider=str(vision_cfg.get("provider", "omniparser")),
            endpoint=self._normalize_parse_endpoint(endpoint),
            probe_endpoint=self._normalize_probe_endpoint(probe_endpoint),
            timeout=int(vision_cfg.get("timeout", 30)),
            retry=int(vision_cfg.get("retry", 2)),
            enabled=bool(vision_cfg.get("enabled", True)),
            autostart=bool(vision_cfg.get("autostart", True)),
            startup_timeout=int(vision_cfg.get("startup_timeout", 180)),
            python_path=str(vision_cfg.get("python_path") or sys.executable),
            project_root=str(vision_cfg.get("project_root") or _DEFAULT_OMNI_RUNTIME),
            caption_model_name=str(vision_cfg.get("caption_model_name") or "florence2"),
            som_model_path=str(
                vision_cfg.get("som_model_path") or _DEFAULT_OMNI_WEIGHTS / "icon_detect" / "model.pt"
            ),
            caption_model_path=str(
                vision_cfg.get("caption_model_path") or _DEFAULT_OMNI_WEIGHTS / "icon_caption_florence"
            ),
            cache_dir=str(vision_cfg.get("cache_dir") or _DEFAULT_OMNI_WEIGHTS / ".cache"),
            include_caption=bool(vision_cfg.get("include_caption", False)),
            include_ocr=bool(vision_cfg.get("include_ocr", False)),
        )

    def _coerce_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _infer_confidence(self, item: dict[str, Any]) -> float:
        if item.get("interactivity") is True:
            return 0.72
        kind = str(item.get("type") or "").lower()
        if any(token in kind for token in ("icon", "button", "input", "field", "list", "card")):
            return 0.66
        return 0.5

    def _normalize_bbox(
        self,
        bbox: list[Any] | tuple[Any, ...],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        values = [float(value) for value in bbox]
        if all(0.0 <= value <= 1.0 for value in values):
            width, height = image_size
            left = int(round(values[0] * width))
            top = int(round(values[1] * height))
            right = int(round(values[2] * width))
            bottom = int(round(values[3] * height))
            return (left, top, right, bottom)
        return (int(round(values[0])), int(round(values[1])), int(round(values[2])), int(round(values[3])))

    def _ensure_service_available(self, requests_module: Any, force_restart: bool = False) -> bool:
        if self._probe(requests_module):
            return True
        if not self._config.autostart:
            return False
        if force_restart:
            self._stop_local_server()
        if not self._start_local_server():
            return False
        deadline = time.time() + max(5, self._config.startup_timeout)
        while time.time() < deadline:
            if self._probe(requests_module):
                return True
            if self._server_process and self._server_process.poll() is not None:
                return False
            time.sleep(1.0)
        return False

    def _probe(self, requests_module: Any) -> bool:
        try:
            response = requests_module.get(
                self._config.probe_endpoint,
                timeout=min(5, self._config.timeout),
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _start_local_server(self) -> bool:
        if self._server_process and self._server_process.poll() is None:
            return True
        # Delegate to WorkerManager for lifecycle management
        try:
            from src.runtime.worker_manager import get_worker_manager
            from src.runtime.paths import RuntimePaths

            vision_cfg = self._raw_vision_cfg or {}
            ocr_cfg = {}
            try:
                from src.common.config_manager import load_config
                cfg = load_config()
                ocr_cfg = {
                    "worker_python": cfg.ocr.worker_python,
                    "persistent_worker_script": cfg.ocr.persistent_worker_script,
                }
            except Exception:
                pass

            paths = RuntimePaths.from_config(vision_cfg=vision_cfg, ocr_cfg=ocr_cfg)
            wm = get_worker_manager(paths)
            host, port = self._parse_host_port(self._config.endpoint)
            worker = wm.start_omniparser(
                preferred_port=port,
                python_path=self._config.python_path,
                startup_timeout=self._config.startup_timeout,
            )
            # Update config endpoint if port changed
            if worker.endpoint and worker.endpoint != self._config.endpoint:
                self._config = RemoteVisionConfig(
                    **{**self._config.__dict__, "endpoint": worker.endpoint,
                       "probe_endpoint": worker.probe_endpoint or self._config.probe_endpoint}
                )
            self._server_process = worker.process
            return True
        except Exception as exc:
            logger.warning("WorkerManager failed to start OmniParser: %s", exc, exc_info=True)
            return False

    def _start_local_server_inline(self) -> bool:
        """Legacy inline startup — kept as reference, not called by default."""
        if self._server_process and self._server_process.poll() is None:
            return True
        python_path = Path(self._config.python_path)
        if not python_path.exists():
            return False
        som_model_path = Path(self._config.som_model_path)
        caption_model_path = Path(self._config.caption_model_path)
        project_root = Path(self._config.project_root)
        if not som_model_path.exists() or not caption_model_path.exists() or not project_root.exists():
            return False
        host, port = self._parse_host_port(self._config.endpoint)
        launch_script = "\n".join(
            [
                "import sys",
                "import types",
                "import importlib",
                "import importlib.machinery",
                "import uvicorn",
                "import torch",
                "from transformers import AutoModelForCausalLM, AutoProcessor",
                "import transformers.utils as tr_utils",
                "from transformers.utils import import_utils as tr_import_utils",
                "import transformers.dynamic_module_utils as tr_dynamic_utils",
                f"sys.path.insert(0, r'{project_root}')",
                f"sys.path.insert(0, r'{project_root / 'omnitool'}')",
                "def _install_flash_attn_stub():",
                "    flash_attn = types.ModuleType('flash_attn')",
                "    flash_attn.__spec__ = importlib.machinery.ModuleSpec('flash_attn', loader=None)",
                "    bert_padding = types.ModuleType('flash_attn.bert_padding')",
                "    bert_padding.__spec__ = importlib.machinery.ModuleSpec('flash_attn.bert_padding', loader=None)",
                "    def _unsupported(*args, **kwargs):",
                "        raise RuntimeError('flash_attn is unavailable on this local OmniParser runtime')",
                "    bert_padding.index_first_axis = _unsupported",
                "    bert_padding.pad_input = _unsupported",
                "    bert_padding.unpad_input = _unsupported",
                "    flash_attn.bert_padding = bert_padding",
                "    sys.modules.setdefault('flash_attn', flash_attn)",
                "    sys.modules.setdefault('flash_attn.bert_padding', bert_padding)",
                "def _patched_check_imports(filename):",
                "    imports = [imp for imp in tr_dynamic_utils.get_imports(filename) if imp != 'flash_attn']",
                "    missing = []",
                "    for imp in imports:",
                "        try:",
                "            importlib.import_module(imp)",
                "        except ImportError:",
                "            missing.append(imp)",
                "    if missing:",
                "        raise ImportError(",
                "            'This modeling file requires the following packages that were not found in your environment: '",
                "            + ', '.join(missing)",
                "            + '. Run `pip install '",
                "            + ' '.join(missing)",
                "            + '`'",
                "        )",
                "    return tr_dynamic_utils.get_relative_imports(filename)",
                "_install_flash_attn_stub()",
                "tr_utils.is_flash_attn_2_available = lambda: False",
                "tr_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False",
                "tr_import_utils.is_flash_attn_2_available = lambda: False",
                "tr_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False",
                "tr_dynamic_utils.check_imports = _patched_check_imports",
                "import util.utils as omn_utils",
                "def _patched_get_caption_model_processor(model_name, model_name_or_path, device=None):",
                "    if not device:",
                "        device = 'cuda' if torch.cuda.is_available() else 'cpu'",
                "    if model_name != 'florence2':",
                "        return omn_utils._original_get_caption_model_processor(model_name, model_name_or_path, device=device)",
                "    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)",
                "    torch_dtype = torch.float32 if device == 'cpu' else torch.float16",
                "    model = AutoModelForCausalLM.from_pretrained(",
                "        model_name_or_path,",
                "        torch_dtype=torch_dtype,",
                "        trust_remote_code=True,",
                "        local_files_only=True,",
                "        attn_implementation='eager',",
                "    ).to(device)",
                "    return {'model': model.to(device), 'processor': processor}",
                "omn_utils._original_get_caption_model_processor = omn_utils.get_caption_model_processor",
                "omn_utils.get_caption_model_processor = _patched_get_caption_model_processor",
                (
                    "sys.argv = ["
                    "'omniparserserver',"
                    f"'--som_model_path', r'{som_model_path}',"
                    f"'--caption_model_name', '{self._config.caption_model_name}',"
                    f"'--caption_model_path', r'{caption_model_path}',"
                    f"'--host', '{host}',"
                    f"'--port', '{port}'"
                    "]"
                ),
                "from omniparserserver.omniparserserver import app",
                f"uvicorn.run(app, host='{host}', port={port}, reload=False)",
            ]
        )
        env = os.environ.copy()
        env.setdefault("HF_HOME", self._config.cache_dir)
        env.setdefault("HUGGINGFACE_HUB_CACHE", self._config.cache_dir)
        env.setdefault("TRANSFORMERS_CACHE", self._config.cache_dir)
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log_dir = Path("data/inspect/runtime")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "omniparser_server.log"
        self._server_log_handle = open(log_path, "a", encoding="utf-8")
        self._server_process = subprocess.Popen(
            [str(python_path), "-c", launch_script],
            cwd=str(project_root),
            env=env,
            stdout=self._server_log_handle,
            stderr=self._server_log_handle,
            text=True,
            creationflags=creationflags,
        )
        return True

    def _stop_local_server(self) -> None:
        try:
            from src.runtime.worker_manager import get_worker_manager
            wm = get_worker_manager()
            wm.stop_worker("omniparser")
        except Exception:
            pass
        # Also handle legacy direct process
        if self._server_process and self._server_process.poll() is None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
        self._server_process = None
        if self._server_log_handle is not None:
            try:
                self._server_log_handle.close()
            except Exception:
                pass
        self._server_log_handle = None

    def _parse_host_port(self, endpoint: str) -> tuple[str, int]:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8001)
        return host, port

    @staticmethod
    def _normalize_parse_endpoint(endpoint: str) -> str:
        value = str(endpoint or "").strip() or "http://127.0.0.1:8001/parse/"
        if value.endswith("/probe") or value.endswith("/probe/"):
            value = value.rstrip("/")[:-5] + "/parse/"
        elif not value.rstrip("/").endswith("/parse"):
            value = value.rstrip("/") + "/parse/"
        elif not value.endswith("/"):
            value += "/"
        return value.replace("localhost", "127.0.0.1")

    @staticmethod
    def _derive_probe_endpoint(parse_endpoint: str) -> str:
        value = OmniParserRemoteVisionProvider._normalize_parse_endpoint(parse_endpoint)
        return value.replace("/parse/", "/probe/")

    @staticmethod
    def _normalize_probe_endpoint(endpoint: str) -> str:
        value = str(endpoint or "").strip()
        if not value:
            return "http://127.0.0.1:8001/probe/"
        if value.rstrip("/").endswith("/parse"):
            value = value.rstrip("/")[:-5] + "/probe/"
        elif not value.rstrip("/").endswith("/probe"):
            value = value.rstrip("/") + "/probe/"
        elif not value.endswith("/"):
            value += "/"
        return value.replace("localhost", "127.0.0.1")
