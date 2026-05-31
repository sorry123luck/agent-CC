"""Configuration manager for DeskCanvas.

Loads settings from YAML config files and environment variables.
Environment variables override file-based config for sensitive values.
"""

from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _ROOT_DIR / "config"
_LOCAL_CONFIG_DIR = _ROOT_DIR / "data" / "local"
_LOCAL_SEMANTIC_MODELER_PATH = _LOCAL_CONFIG_DIR / "semantic_modeler.json"


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the config directory."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}


def _env_or(key: str, default: str = "") -> str:
    """Read environment variable, falling back to default."""
    return os.environ.get(key, default)


def _env_bool_or(key: str, default: bool) -> bool:
    """Read a boolean environment variable when explicitly set."""
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON from a local config file."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_local_semantic_modeler() -> dict[str, Any]:
    """Load local runtime semantic modeler config."""
    return _load_json(_LOCAL_SEMANTIC_MODELER_PATH)


@dataclass(frozen=True)
class VLMConfig:
    """VLM provider configuration."""
    provider: str = "disabled"  # cloud / local / disabled
    api_key: str = ""
    endpoint: str = ""
    model: str = "gpt-4o"
    timeout_seconds: int = 30
    max_retries: int = 1
    proxy_url: str = ""
    proxy_port: int = 0
    fixed_anchor_skip_interval: int = 3  # Phase 6: VLM call frequency for fixed_anchor candidates


@dataclass(frozen=True)
class OCRConfig:
    """OCR provider configuration."""
    provider: str = "paddleocr_bridge"
    lang: str = "ch"
    bridge_enabled: bool = True
    worker_python: str = ""
    worker_script: str = ""
    persistent_worker_script: str = ""
    worker_mode: str = "persistent"
    worker_timeout_seconds: int = 120


@dataclass(frozen=True)
class VisionConfig:
    """Vision provider configuration."""
    provider: str = "omniparser"
    endpoint: str = ""
    timeout: int = 90
    retry: int = 2


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy settings."""
    save_screenshots: bool = False
    auto_cleanup_hours: int = 24
    redact_sensitive: bool = True
    default_redaction_level: str = "memory"


@dataclass(frozen=True)
class ServerConfig:
    """API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class CacheConfig:
    """Canvas cache configuration."""
    l1_max_size: int = 8        # 内存 LRU 容量 [1, 64]
    l2_max_files: int = 300     # 磁盘 warm cache 最大文件数
    l2_jpeg_quality: int = 80   # JPEG quality [75, 85]
    l3_keyframe_quality: int = 85  # L3 关键帧 quality（Phase 5 用）


@dataclass(frozen=True)
class SemanticModelerConfig:
    """VLM Semantic Modeler configuration."""
    enabled: bool = False
    provider: str = "disabled"      # openai / anthropic / minimax / mock / disabled
    api_key: str = ""
    endpoint: str = ""
    model: str = ""
    fallback_provider: str = ""
    fallback_model: str = ""
    provider_variant: str = ""      # OpenRouter 等兼容 API 的 variant 标识
    free_model_only: bool = False
    allow_model_fallback: bool = True
    timeout_seconds: int = 60
    max_retries: int = 2
    proxy_url: str = ""              # Empty means direct transport.
    proxy_port: int = 0              # Optional localhost proxy port, e.g. 7890.
    prompt_version: str = "1.0"
    # 额度/成本控制
    daily_call_limit: int = 100
    monthly_budget_usd: float = 10.0
    priority: int = 1
    allow_free_models: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    # 隐私
    save_raw_response: bool = False
    redact_dynamic_content: bool = True
    # 推理模式
    thinking_mode: str = "auto"   # auto / on / off
    # 图片压缩
    image_max_width: int = 1280   # 超过此宽度等比缩放


@dataclass(frozen=True)
class GeometricPartitionerConfig:
    """Geometric partitioner thresholds (from config/vision.yaml)."""
    run_legacy_visual_layout: bool = False
    separator_threshold: float = 10.0
    min_span_ratio: float = 0.55
    density_contrast_threshold: float = 0.12
    min_region_area_ratio: float = 0.006
    border_edge_threshold: int = 12
    floating_min_confidence: float = 0.65
    grid_min_card_count: int = 4
    content_std_min: float = 6.0
    content_std_max: float = 42.0
    low_texture_std_max: float = 4.0


@dataclass(frozen=True)
class SemanticFusionConfig:
    """Semantic fusion thresholds (from config/vision.yaml)."""
    uia_input_confidence: float = 0.85
    action_bar_min_buttons: int = 3
    action_bar_band_ratio: float = 0.22
    navigation_min_items: int = 3
    navigation_side_band_ratio: float = 0.30
    content_min_area_ratio: float = 0.40
    unknown_threshold: float = 0.35


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    vlm: VLMConfig = field(default_factory=VLMConfig)
    semantic_modeler: SemanticModelerConfig = field(default_factory=SemanticModelerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    geometric_partitioner: GeometricPartitionerConfig = field(default_factory=GeometricPartitionerConfig)
    semantic_fusion: SemanticFusionConfig = field(default_factory=SemanticFusionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    debug: bool = False


_SEMANTIC_MODELER_FIELDS = set(SemanticModelerConfig.__dataclass_fields__)


def semantic_modeler_to_public_dict(config: SemanticModelerConfig | None = None) -> dict[str, Any]:
    """Return semantic modeler config without exposing the raw API key."""
    cfg = config or load_config().semantic_modeler
    data = asdict(cfg)
    api_key = data.pop("api_key", "")
    data["api_key_set"] = bool(api_key)
    return data


def save_local_semantic_modeler_config(updates: dict[str, Any]) -> SemanticModelerConfig:
    """Persist local semantic modeler settings.

    The API key is preserved when omitted, updated when present, and cleared when
    explicitly set to an empty string.
    """
    current = _load_local_semantic_modeler()
    next_config = {
        k: v
        for k, v in current.items()
        if k in _SEMANTIC_MODELER_FIELDS
    }

    for key, value in updates.items():
        if key not in _SEMANTIC_MODELER_FIELDS or value is None:
            continue
        next_config[key] = value

    _LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOCAL_SEMANTIC_MODELER_PATH, "w", encoding="utf-8") as f:
        json.dump(next_config, f, ensure_ascii=False, indent=2)

    return load_config().semantic_modeler


def load_config() -> AppConfig:
    """Load the full application configuration.

    Priority: environment variables > YAML config > defaults.
    """
    app_yaml = _load_yaml("app.yaml")
    models_yaml = _load_yaml("models.yaml")

    # models.yaml may nest under "models:" key or be flat
    models_root = models_yaml.get("models", models_yaml)

    # Server config
    api_section = app_yaml.get("api", {})
    server = ServerConfig(
        host=_env_or("OPENCLAW_HOST", api_section.get("host", "127.0.0.1")),
        port=int(_env_or("OPENCLAW_PORT", str(api_section.get("port", 8000)))),
    )

    # VLM config (from cloud_vision in models.yaml or env)
    cloud = models_root.get("cloud_vision", {})
    vlm_env_provider = _env_or("OPENCLAW_VLM_PROVIDER", "")
    vlm = VLMConfig(
        provider=vlm_env_provider or ("cloud" if cloud.get("enabled") else "disabled"),
        api_key=_env_or(cloud.get("api_key_env", ""), cloud.get("api_key", "")),
        endpoint=_env_or("OPENCLAW_VLM_ENDPOINT", cloud.get("endpoint", "")),
        model=_env_or("OPENCLAW_VLM_MODEL", cloud.get("model", "")),
        proxy_url=_env_or("OPENCLAW_VLM_PROXY_URL", cloud.get("proxy_url", "")),
        proxy_port=int(_env_or("OPENCLAW_VLM_PROXY_PORT", str(cloud.get("proxy_port", 0) or 0))),
    )

    # OCR config
    ocr_section = models_root.get("ocr", {})
    ocr = OCRConfig(
        provider=ocr_section.get("provider", "paddleocr_bridge"),
        lang=ocr_section.get("lang", "ch"),
        bridge_enabled=ocr_section.get("bridge_enabled", True),
        worker_python=ocr_section.get("worker_python", ""),
        worker_script=ocr_section.get("worker_script", ""),
        persistent_worker_script=ocr_section.get("persistent_worker_script", ""),
        worker_mode=ocr_section.get("worker_mode", "persistent"),
        worker_timeout_seconds=ocr_section.get("worker_timeout_seconds", 120),
    )

    # Vision config
    vision_section = models_root.get("vision", {})
    vision = VisionConfig(
        provider=vision_section.get("provider", "omniparser"),
        endpoint=_env_or("OPENCLAW_VISION_ENDPOINT", vision_section.get("endpoint", "")),
        timeout=vision_section.get("timeout", 90),
        retry=vision_section.get("retry", 2),
    )

    # Privacy config
    privacy = PrivacyConfig(
        save_screenshots=_env_or("OPENCLAW_SAVE_SCREENSHOTS", "false").lower() == "true",
        redact_sensitive=_env_or("OPENCLAW_REDACT_SENSITIVE", "true").lower() != "false",
    )

    # Semantic Modeler config (from models.yaml, local runtime config, or env)
    sm_section = {
        **models_root.get("semantic_modeler", {}),
        **_load_local_semantic_modeler(),
    }
    sm_env_provider = _env_or("OPENCLAW_SEMANTIC_MODELER_PROVIDER", "")
    semantic_modeler = SemanticModelerConfig(
        enabled=_env_bool_or("OPENCLAW_SEMANTIC_MODELER_ENABLED", sm_section.get("enabled", False)),
        provider=sm_env_provider or sm_section.get("provider", "disabled"),
        api_key=_env_or("OPENCLAW_SEMANTIC_MODELER_API_KEY", sm_section.get("api_key", "")),
        endpoint=_env_or("OPENCLAW_SEMANTIC_MODELER_ENDPOINT", sm_section.get("endpoint", "")),
        model=_env_or("OPENCLAW_SEMANTIC_MODELER_MODEL", sm_section.get("model", "")),
        fallback_provider=sm_section.get("fallback_provider", ""),
        fallback_model=sm_section.get("fallback_model", ""),
        provider_variant=_env_or("OPENCLAW_SEMANTIC_MODELER_PROVIDER_VARIANT", sm_section.get("provider_variant", "")),
        free_model_only=_env_bool_or("OPENCLAW_SEMANTIC_MODELER_FREE_MODEL_ONLY", sm_section.get("free_model_only", False)),
        allow_model_fallback=sm_section.get("allow_model_fallback", True),
        timeout_seconds=sm_section.get("timeout_seconds", 60),
        max_retries=sm_section.get("max_retries", 2),
        proxy_url=_env_or("OPENCLAW_SEMANTIC_MODELER_PROXY_URL", sm_section.get("proxy_url", "")),
        proxy_port=int(_env_or("OPENCLAW_SEMANTIC_MODELER_PROXY_PORT", str(sm_section.get("proxy_port", 0) or 0))),
        prompt_version=sm_section.get("prompt_version", "1.0"),
        daily_call_limit=int(_env_or("OPENCLAW_SEMANTIC_MODELER_DAILY_CALL_LIMIT", str(sm_section.get("daily_call_limit", 100)))),
        monthly_budget_usd=float(_env_or("OPENCLAW_SEMANTIC_MODELER_MONTHLY_BUDGET_USD", str(sm_section.get("monthly_budget_usd", 10.0)))),
        priority=sm_section.get("priority", 1),
        allow_free_models=_env_bool_or("OPENCLAW_SEMANTIC_MODELER_ALLOW_FREE_MODELS", sm_section.get("allow_free_models", True)),
        cost_per_1k_input=float(sm_section.get("cost_per_1k_input", 0.0)),
        cost_per_1k_output=float(sm_section.get("cost_per_1k_output", 0.0)),
        save_raw_response=_env_bool_or("OPENCLAW_SEMANTIC_MODELER_SAVE_RAW_RESPONSE", sm_section.get("save_raw_response", False)),
        redact_dynamic_content=_env_bool_or("OPENCLAW_SEMANTIC_MODELER_REDACT_DYNAMIC_CONTENT", sm_section.get("redact_dynamic_content", True)),
        thinking_mode=sm_section.get("thinking_mode", "auto"),
        image_max_width=int(sm_section.get("image_max_width", 1280)),
    )

    debug = app_yaml.get("app", {}).get("debug", False) or _env_or("OPENCLAW_DEBUG", "").lower() == "true"

    # Cache config
    cache_section = app_yaml.get("cache", {})
    cache = CacheConfig(
        l1_max_size=max(1, min(64, int(cache_section.get("l1_max_size", 8)))),
        l2_max_files=max(1, int(cache_section.get("l2_max_files", 300))),
        l2_jpeg_quality=max(75, min(85, int(cache_section.get("l2_jpeg_quality", 80)))),
        l3_keyframe_quality=max(75, min(85, int(cache_section.get("l3_keyframe_quality", 85)))),
    )

    # Geometric partitioner config (from vision.yaml)
    vision_yaml = _load_yaml("vision.yaml")
    gp_section = vision_yaml.get("geometric_partitioner", {})
    geometric_partitioner = GeometricPartitionerConfig(
        run_legacy_visual_layout=bool(gp_section.get("run_legacy_visual_layout", False)),
        separator_threshold=float(gp_section.get("separator_threshold", 10.0)),
        min_span_ratio=float(gp_section.get("min_span_ratio", 0.55)),
        density_contrast_threshold=float(gp_section.get("density_contrast_threshold", 0.12)),
        min_region_area_ratio=float(gp_section.get("min_region_area_ratio", 0.006)),
        border_edge_threshold=int(gp_section.get("border_edge_threshold", 12)),
        floating_min_confidence=float(gp_section.get("floating_min_confidence", 0.65)),
        grid_min_card_count=int(gp_section.get("grid_min_card_count", 4)),
        content_std_min=float(gp_section.get("content_std_min", 6.0)),
        content_std_max=float(gp_section.get("content_std_max", 42.0)),
        low_texture_std_max=float(gp_section.get("low_texture_std_max", 4.0)),
    )

    # Semantic fusion config (from vision.yaml)
    sf_section = vision_yaml.get("semantic_fusion", {})
    semantic_fusion = SemanticFusionConfig(
        uia_input_confidence=float(sf_section.get("uia_input_confidence", 0.85)),
        action_bar_min_buttons=int(sf_section.get("action_bar_min_buttons", 3)),
        action_bar_band_ratio=float(sf_section.get("action_bar_band_ratio", 0.22)),
        navigation_min_items=int(sf_section.get("navigation_min_items", 3)),
        navigation_side_band_ratio=float(sf_section.get("navigation_side_band_ratio", 0.30)),
        content_min_area_ratio=float(sf_section.get("content_min_area_ratio", 0.40)),
        unknown_threshold=float(sf_section.get("unknown_threshold", 0.35)),
    )

    return AppConfig(
        vlm=vlm,
        semantic_modeler=semantic_modeler,
        ocr=ocr,
        vision=vision,
        geometric_partitioner=geometric_partitioner,
        semantic_fusion=semantic_fusion,
        privacy=privacy,
        server=server,
        cache=cache,
        debug=debug,
    )
