"""Unit tests for WorkerManager and RuntimePaths."""

from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.runtime.paths import RuntimePaths, _REPO_DIR
from src.runtime.worker_manager import ManagedWorker, WorkerManager


class TestRuntimePaths:
    def test_from_config_defaults(self):
        paths = RuntimePaths.from_config()
        assert paths.repo_root.exists()
        assert paths.data_dir == paths.repo_root / "data"
        assert paths.worker_script == paths.repo_root / "scripts" / "omniparser_worker.py"
        assert paths.omniparser_log_dir.name == "runtime"

    def test_repo_root_is_always_git_root(self):
        """Even when vision_cfg.project_root points to OmniParser runtime,
        repo_root must remain the git repository root."""
        paths = RuntimePaths.from_config(
            vision_cfg={"project_root": "D:\\vendor\\omniparser_runtime"},
        )
        assert paths.repo_root == _REPO_DIR
        assert "omniparser_runtime" not in str(paths.repo_root)

    def test_worker_script_under_repo_not_omniparser_runtime(self):
        """worker_script must be under repo_root/scripts/, not under omniparser_runtime."""
        paths = RuntimePaths.from_config(
            vision_cfg={"project_root": "D:\\vendor\\omniparser_runtime"},
        )
        assert paths.worker_script == _REPO_DIR / "scripts" / "omniparser_worker.py"
        assert "omniparser_runtime" not in str(paths.worker_script)

    def test_omniparser_runtime_from_config(self):
        paths = RuntimePaths.from_config(
            vision_cfg={"project_root": "D:\\vendor\\omniparser_runtime"},
        )
        assert str(paths.omniparser_runtime) == "D:\\vendor\\omniparser_runtime"

    def test_som_model_path_from_config(self):
        custom = "D:\\models\\omniparser\\weights\\icon_detect\\model.pt"
        paths = RuntimePaths.from_config(
            vision_cfg={"som_model_path": custom},
        )
        assert str(paths.som_model_path) == custom

    def test_omniparser_weights_derived_correctly(self):
        """weights root should be .../weights, not .../weights/icon_detect."""
        custom = "D:\\models\\omniparser\\weights\\icon_detect\\model.pt"
        paths = RuntimePaths.from_config(
            vision_cfg={"som_model_path": custom},
        )
        # weights root = parent of icon_detect = .../weights
        assert paths.omniparser_weights.name == "weights"
        assert paths.omniparser_weights == Path("D:\\models\\omniparser\\weights")

    def test_som_model_path_no_double_icon_detect(self):
        """som_model_path should NOT be re-joined with icon_detect/model.pt."""
        custom = "D:\\models\\omniparser\\weights\\icon_detect\\model.pt"
        paths = RuntimePaths.from_config(
            vision_cfg={"som_model_path": custom},
        )
        # The path should end with icon_detect/model.pt exactly once
        parts = paths.som_model_path.parts
        icon_detect_count = sum(1 for p in parts if p == "icon_detect")
        assert icon_detect_count == 1, f"icon_detect appears {icon_detect_count} times in {paths.som_model_path}"

    def test_caption_model_path_from_config(self):
        custom = "D:\\models\\omniparser\\weights\\icon_caption_florence"
        paths = RuntimePaths.from_config(
            vision_cfg={"caption_model_path": custom},
        )
        assert str(paths.caption_model_path) == custom

    def test_cache_dir_from_config(self):
        custom = "D:\\models\\omniparser\\weights\\.cache"
        paths = RuntimePaths.from_config(
            vision_cfg={"cache_dir": custom},
        )
        assert str(paths.cache_dir) == custom

    def test_ocr_python_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_OCR_PYTHON", "/custom/python")
        paths = RuntimePaths.from_config()
        assert paths.ocr_python == Path("/custom/python")

    def test_ocr_python_from_config(self):
        paths = RuntimePaths.from_config(
            ocr_cfg={"worker_python": "/usr/bin/python3"},
        )
        assert paths.ocr_python == Path("/usr/bin/python3")


class TestManagedWorker:
    def test_defaults(self):
        w = ManagedWorker(provider_id="test")
        assert w.process is None
        assert w.port is None
        assert w.endpoint is None
        assert w.consecutive_failures == 0
        assert isinstance(w._stdout_queue, queue.Queue)


class TestWorkerManagerFindPort:
    def test_find_available_port_returns_preferred_when_free(self):
        wm = WorkerManager(RuntimePaths.from_config())
        port = wm._find_available_port(19999)
        assert port == 19999

    def test_find_available_port_falls_back(self):
        wm = WorkerManager(RuntimePaths.from_config())
        port = wm._find_available_port(1)
        assert port is not None
        assert port >= 1


class TestWorkerManagerLifecycle:
    def test_get_worker_returns_none_when_not_started(self):
        wm = WorkerManager(RuntimePaths.from_config())
        assert wm.get_worker("nonexistent") is None

    def test_is_alive_returns_false_when_no_worker(self):
        wm = WorkerManager(RuntimePaths.from_config())
        assert wm.is_alive("nonexistent") is False

    def test_stop_worker_is_noop_when_not_started(self):
        wm = WorkerManager(RuntimePaths.from_config())
        wm.stop_worker("nonexistent")

    def test_stop_all_is_noop_when_empty(self):
        wm = WorkerManager(RuntimePaths.from_config())
        wm.stop_all()


class TestWorkerManagerStartOmniparser:
    def test_raises_when_runtime_missing(self, tmp_path):
        paths = RuntimePaths(
            repo_root=tmp_path,
            data_dir=tmp_path / "data",
            logs_dir=tmp_path / "data" / "logs",
            temp_dir=tmp_path / "data" / "temp",
            ocr_python=Path("/fake/python"),
            ocr_persistent_script=tmp_path / "fake_script.py",
            worker_script=tmp_path / "scripts" / "omniparser_worker.py",
            omniparser_runtime=tmp_path / "nonexistent_runtime",
            omniparser_weights=tmp_path / "models",
            omniparser_log_dir=tmp_path / "data" / "logs",
            som_model_path=tmp_path / "model.pt",
            caption_model_path=tmp_path / "caption",
            cache_dir=tmp_path / "cache",
        )
        wm = WorkerManager(paths)
        with pytest.raises(FileNotFoundError, match="runtime"):
            wm.start_omniparser()

    def test_raises_when_worker_script_missing(self, tmp_path):
        runtime_dir = tmp_path / "vendor" / "omniparser_runtime"
        runtime_dir.mkdir(parents=True)
        paths = RuntimePaths(
            repo_root=tmp_path,
            data_dir=tmp_path / "data",
            logs_dir=tmp_path / "data" / "logs",
            temp_dir=tmp_path / "data" / "temp",
            ocr_python=Path("/fake/python"),
            ocr_persistent_script=tmp_path / "fake_script.py",
            worker_script=tmp_path / "scripts" / "omniparser_worker.py",
            omniparser_runtime=runtime_dir,
            omniparser_weights=tmp_path / "models",
            omniparser_log_dir=tmp_path / "data" / "logs",
            som_model_path=tmp_path / "model.pt",
            caption_model_path=tmp_path / "caption",
            cache_dir=tmp_path / "cache",
        )
        wm = WorkerManager(paths)
        with pytest.raises(FileNotFoundError, match="worker script"):
            wm.start_omniparser()


class TestStdoutReader:
    def test_persistent_reader_thread(self, tmp_path):
        """Verify _start_stdout_reader creates one thread, not per-readline threads."""
        paths = RuntimePaths.from_config()
        wm = WorkerManager(paths)
        worker = ManagedWorker(provider_id="test")

        # Simulate a process with stdout
        import subprocess
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.stdout = iter(["line1\n", "line2\n", ""])
        worker.process = proc

        wm._start_stdout_reader(worker)
        assert worker._stdout_thread is not None
        assert worker._stdout_thread.daemon is True

        # Should read from queue, not create new threads
        line = wm._readline_from_queue(worker, timeout=2.0)
        assert line == "line1\n"
        line = wm._readline_from_queue(worker, timeout=2.0)
        assert line == "line2\n"

    def test_readline_returns_none_on_timeout(self):
        """_readline_from_queue returns None when queue is empty."""
        paths = RuntimePaths.from_config()
        wm = WorkerManager(paths)
        worker = ManagedWorker(provider_id="test")
        worker._stdout_queue = queue.Queue()
        result = wm._readline_from_queue(worker, timeout=0.01)
        assert result is None


class TestProbeWait:
    def test_probe_returns_true_when_server_responds(self):
        """_wait_for_probe returns True when HTTP probe succeeds."""
        paths = RuntimePaths.from_config()
        wm = WorkerManager(paths)

        with patch("src.runtime.worker_manager.time") as mock_time:
            # First call: before deadline
            mock_time.perf_counter.side_effect = [0.0, 0.0, 0.0]
            mock_time.sleep = lambda x: None

            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.ok = True
                mock_get.return_value = mock_resp
                result = wm._wait_for_probe("http://127.0.0.1:9999/probe", timeout=5)
                assert result is True

    def test_probe_returns_false_on_timeout(self):
        """_wait_for_probe returns False when deadline exceeded."""
        paths = RuntimePaths.from_config()
        wm = WorkerManager(paths)

        with patch("src.runtime.worker_manager.time") as mock_time:
            # Simulate: first perf_counter=0, then always past deadline
            call_count = [0]
            def fake_perf():
                call_count[0] += 1
                if call_count[0] <= 1:
                    return 0.0
                return 100.0  # way past deadline
            mock_time.perf_counter.side_effect = fake_perf
            mock_time.sleep = lambda x: None

            with patch("requests.get", side_effect=Exception("no server")):
                result = wm._wait_for_probe("http://127.0.0.1:9999/probe", timeout=1)
                assert result is False


class TestWorkerManagerSingleton:
    def test_singleton(self):
        from src.runtime.worker_manager import get_worker_manager
        a = get_worker_manager()
        b = get_worker_manager()
        assert a is b
