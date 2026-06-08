"""Unit tests for config_manager — verifies YAML read path and env overrides."""

from __future__ import annotations

import os
from unittest.mock import patch
import json

from src.common.config_manager import load_config


def _fake_models_yaml(data: dict):
    """Return a patch for _load_yaml that returns data for models.yaml."""
    def _loader(filename: str):
        if filename == "models.yaml":
            return data
        return {}
    return _loader


class TestConfigManagerReadPath:
    """Verify config_manager reads from models.models.cloud_vision correctly."""

    def test_load_config_reads_cloud_vision_enabled(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "model": "gpt-4o"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.vlm.provider == "cloud"

    def test_load_config_reads_cloud_vision_disabled(self):
        yaml = {"models": {"cloud_vision": {"enabled": False}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.vlm.provider == "disabled"

    def test_load_config_reads_cloud_vision_api_key_env(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "api_key_env": "TEST_VLM_KEY"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            with patch.dict(os.environ, {"TEST_VLM_KEY": "sk-test-123"}):
                cfg = load_config()
        assert cfg.vlm.api_key == "sk-test-123"

    def test_load_config_reads_cloud_vision_model_from_config(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "model": "claude-sonnet-4-20250514"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.vlm.model == "claude-sonnet-4-20250514"

    def test_load_config_reads_vision_endpoint(self):
        yaml = {"models": {"vision": {"provider": "omniparser", "endpoint": "http://localhost:9001/parse/"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.vision.endpoint == "http://localhost:9001/parse/"

    def test_load_config_reads_ocr_provider(self):
        yaml = {"models": {"ocr": {"provider": "paddleocr_bridge", "lang": "en"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.ocr.provider == "paddleocr_bridge"
        assert cfg.ocr.lang == "en"

    def test_vlm_provider_env_overrides_config(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "model": "gpt-4o"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            with patch.dict(os.environ, {"OPENCLAW_VLM_PROVIDER": "local"}):
                cfg = load_config()
        assert cfg.vlm.provider == "local"

    def test_vlm_model_env_overrides_config(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "model": "gpt-4o"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            with patch.dict(os.environ, {"OPENCLAW_VLM_MODEL": "qwen-vl-max"}):
                cfg = load_config()
        assert cfg.vlm.model == "qwen-vl-max"

    def test_vlm_proxy_defaults_to_direct(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "model": "gpt-4o"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.vlm.proxy_url == ""
        assert cfg.vlm.proxy_port == 0

    def test_vlm_proxy_port_env_overrides_config(self):
        yaml = {"models": {"cloud_vision": {"enabled": True, "proxy_port": 7890}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            with patch.dict(os.environ, {"OPENCLAW_VLM_PROXY_PORT": "10808"}):
                cfg = load_config()
        assert cfg.vlm.proxy_port == 10808

    def test_flat_yaml_without_models_key(self):
        """Support flat YAML without 'models:' wrapper."""
        yaml = {"cloud_vision": {"enabled": True, "model": "gpt-4o-mini"}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            cfg = load_config()
        assert cfg.vlm.provider == "cloud"
        assert cfg.vlm.model == "gpt-4o-mini"

    def test_empty_yaml_returns_defaults(self):
        with patch("src.common.config_manager._load_yaml", return_value={}):
            cfg = load_config()
        assert cfg.vlm.provider == "disabled"
        assert cfg.vlm.model == ""
        assert cfg.vision.provider == "omniparser"
        assert cfg.ocr.provider == "paddleocr_bridge"

    def test_local_semantic_modeler_json_allows_utf8_bom(self, tmp_path):
        local_path = tmp_path / "semantic_modeler.json"
        local_path.write_text(
            "\ufeff" + json.dumps({
                "enabled": True,
                "provider": "qwen",
                "model": "qwen3-vl-flash",
                "api_key": "sk-test",
            }),
            encoding="utf-8",
        )
        yaml = {"models": {"semantic_modeler": {"enabled": False, "provider": "disabled"}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            with patch("src.common.config_manager._LOCAL_SEMANTIC_MODELER_PATH", local_path):
                cfg = load_config()

        assert cfg.semantic_modeler.enabled is True
        assert cfg.semantic_modeler.provider == "qwen"
        assert cfg.semantic_modeler.model == "qwen3-vl-flash"

    def test_semantic_modeler_proxy_port_env_overrides_config(self):
        yaml = {"models": {"semantic_modeler": {"enabled": True, "proxy_port": 7890}}}
        with patch("src.common.config_manager._load_yaml", side_effect=_fake_models_yaml(yaml)):
            with patch.dict(os.environ, {"OPENCLAW_SEMANTIC_MODELER_PROXY_PORT": "10808"}):
                cfg = load_config()

        assert cfg.semantic_modeler.proxy_url == ""
        assert cfg.semantic_modeler.proxy_port == 10808
