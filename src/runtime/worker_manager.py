"""Worker manager — lifecycle management for subprocess-based providers.

Currently manages:
- OmniParser (uvicorn subprocess with flash_attn patches)

Future:
- PaddleOCR persistent worker (if unified lifecycle is desired)
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.runtime.paths import RuntimePaths

logger = logging.getLogger(__name__)


@dataclass
class ManagedWorker:
    """State of a managed subprocess worker."""

    provider_id: str
    process: subprocess.Popen | None = None
    port: int | None = None
    endpoint: str | None = None
    probe_endpoint: str | None = None
    started_at: float = 0.0
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    log_path: str | None = None
    _log_handle: Any = field(default=None, repr=False)
    _stdout_queue: queue.Queue[str] = field(default_factory=queue.Queue, repr=False)
    _stdout_thread: threading.Thread | None = field(default=None, repr=False)


class WorkerManager:
    """Manage subprocess-based provider lifecycles.

    Responsibilities:
    - Start OmniParser with auto-port selection
    - Wait for ready event from worker stdout
    - Probe HTTP endpoint before confirming readiness
    - Track worker health
    - Graceful shutdown on process exit

    Does NOT:
    - Change observe/query behavior
    - Auto-restart crashed workers (that's Phase C fallback policy)
    """

    def __init__(self, paths: RuntimePaths) -> None:
        self._paths = paths
        self._workers: dict[str, ManagedWorker] = {}
        self._lock = threading.Lock()

    def get_worker(self, provider_id: str) -> ManagedWorker | None:
        """Return the managed worker for *provider_id*, or None."""
        return self._workers.get(provider_id)

    def start_omniparser(
        self,
        *,
        preferred_port: int = 8001,
        python_path: str | None = None,
        startup_timeout: int = 180,
    ) -> ManagedWorker:
        """Start OmniParser subprocess with auto-port selection.

        1. Find an available port (preferred_port first, then scan 18000-18100)
        2. Launch scripts/omniparser_worker.py
        3. Wait for {"event": "ready", "port": N} from stdout
        4. HTTP probe the /probe endpoint until it responds
        5. Return ManagedWorker with confirmed endpoint
        """
        with self._lock:
            existing = self._workers.get("omniparser")
            if existing and existing.process and existing.process.poll() is None:
                logger.info("OmniParser already running on port %s", existing.port)
                return existing

        port = self._find_available_port(preferred_port)
        if port is None:
            raise RuntimeError(
                f"No available port for OmniParser (tried {preferred_port} + 18000-18100)"
            )

        py = python_path or self._resolve_omniparser_python()

        # Use paths fields directly — no re-derivation
        project_root = self._paths.omniparser_runtime
        worker_script = self._paths.worker_script

        if not project_root.exists():
            raise FileNotFoundError(f"OmniParser runtime not found: {project_root}")
        if not worker_script.exists():
            raise FileNotFoundError(f"OmniParser worker script not found: {worker_script}")

        # Direct config paths — no re-joining
        som_model = str(self._paths.som_model_path)
        caption_model = str(self._paths.caption_model_path)
        cache_dir = str(self._paths.cache_dir)

        # Ensure log directory exists
        log_dir = self._paths.omniparser_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "omniparser_server.log"
        log_handle = open(log_path, "a", encoding="utf-8")

        cmd = [
            py, str(worker_script),
            "--project-root", str(project_root),
            "--som-model-path", som_model,
            "--caption-model-name", "florence2",
            "--caption-model-path", caption_model,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--cache-dir", cache_dir,
        ]

        env = self._build_omniparser_env(cache_dir)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        logger.info("Starting OmniParser on port %d (python=%s)", port, py)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )

        worker = ManagedWorker(
            provider_id="omniparser",
            process=proc,
            port=port,
            endpoint=f"http://127.0.0.1:{port}/parse/",
            probe_endpoint=f"http://127.0.0.1:{port}/probe/",
            started_at=time.perf_counter(),
            log_path=str(log_path),
            _log_handle=log_handle,
        )

        # Start persistent stdout reader thread
        self._start_stdout_reader(worker)

        # Wait for ready event
        actual_port = self._wait_for_ready(worker, startup_timeout)
        if actual_port is None:
            self._kill_worker(worker)
            raise TimeoutError(f"OmniParser did not emit ready within {startup_timeout}s")

        if actual_port != port:
            worker.port = actual_port
            worker.endpoint = f"http://127.0.0.1:{actual_port}/parse/"
            worker.probe_endpoint = f"http://127.0.0.1:{actual_port}/probe/"

        # Confirm HTTP probe is reachable
        probe_ok = self._wait_for_probe(worker.probe_endpoint, timeout=30)
        if not probe_ok:
            self._kill_worker(worker)
            raise TimeoutError(
                f"OmniParser probe not reachable at {worker.probe_endpoint} within 30s after ready"
            )

        with self._lock:
            self._workers["omniparser"] = worker

        elapsed = time.perf_counter() - worker.started_at
        logger.info("OmniParser ready on port %d (%.1fs, probe confirmed)", actual_port, elapsed)
        return worker

    def stop_worker(self, provider_id: str) -> None:
        """Gracefully stop a managed worker."""
        with self._lock:
            worker = self._workers.pop(provider_id, None)
        if worker:
            self._kill_worker(worker)

    def stop_all(self) -> None:
        """Stop all managed workers (called at process exit)."""
        with self._lock:
            ids = list(self._workers.keys())
        for pid in ids:
            self.stop_worker(pid)

    def is_alive(self, provider_id: str) -> bool:
        """Check if the worker process is still running."""
        worker = self._workers.get(provider_id)
        if not worker or not worker.process:
            return False
        return worker.process.poll() is None

    # -- internals -----------------------------------------------------------

    def _find_available_port(self, preferred: int) -> int | None:
        """Try preferred port first, then scan 18000-18100."""
        for port in [preferred, *range(18000, 18100)]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return None

    def _resolve_omniparser_python(self) -> str:
        """Resolve Python path for OmniParser from config."""
        try:
            from src.common.config_manager import load_config

            cfg = load_config()
            py = getattr(cfg.vision, "python_path", "")
            if py and Path(py).exists():
                return py
        except Exception:
            pass
        return sys.executable

    def _build_omniparser_env(self, cache_dir: str) -> dict[str, str]:
        """Build environment for OmniParser subprocess."""
        import os

        env = os.environ.copy()
        env.setdefault("HF_HOME", cache_dir)
        env.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
        env.setdefault("TRANSFORMERS_CACHE", cache_dir)
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        return env

    # -- stdout reader (persistent thread) -----------------------------------

    def _start_stdout_reader(self, worker: ManagedWorker) -> None:
        """Start a single persistent thread that reads stdout into a queue."""
        proc = worker.process
        if not proc or not proc.stdout:
            return

        def _reader():
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    worker._stdout_queue.put(line)
            except Exception:
                pass
            # Signal EOF
            worker._stdout_queue.put("")

        t = threading.Thread(target=_reader, daemon=True, name=f"omniparser-stdout-{worker.port}")
        t.start()
        worker._stdout_thread = t

    def _readline_from_queue(self, worker: ManagedWorker, timeout: float) -> str | None:
        """Read one line from the persistent stdout queue with timeout."""
        try:
            return worker._stdout_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- ready / probe waiting -----------------------------------------------

    def _wait_for_ready(self, worker: ManagedWorker, timeout: int) -> int | None:
        """Read stdout until {"event": "ready", "port": N} or timeout."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            proc = worker.process
            if proc and proc.poll() is not None:
                logger.error("OmniParser process exited during startup (code=%s)", proc.returncode)
                return None

            remaining = max(0.1, deadline - time.perf_counter())
            line = self._readline_from_queue(worker, min(remaining, 1.0))
            if line is None:
                continue
            line = line.strip()
            if not line:
                # EOF
                if worker.process and worker.process.poll() is not None:
                    return None
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("OmniParser stdout (non-JSON): %s", line[:200])
                continue

            if event.get("event") == "ready":
                return int(event.get("port") or worker.port)
            if event.get("event") == "model_loaded":
                logger.info("OmniParser model loaded: %s", event)
                continue
            logger.debug("OmniParser event: %s", event)

        return None

    def _wait_for_probe(self, probe_endpoint: str, timeout: float = 30) -> bool:
        """HTTP GET probe until it responds 2xx or timeout."""
        try:
            import requests as _requests
        except ImportError:
            logger.warning("requests not installed, skipping probe check")
            return True

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                resp = _requests.get(probe_endpoint, timeout=3.0)
                if resp.ok:
                    return True
            except Exception:
                pass
            time.sleep(1.0)

        return False

    def _kill_worker(self, worker: ManagedWorker) -> None:
        """Terminate/kill a worker process."""
        proc = worker.process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if worker._log_handle:
            try:
                worker._log_handle.close()
            except Exception:
                pass
        worker.process = None
        logger.info("OmniParser worker stopped (port=%s)", worker.port)


# Singleton
_worker_manager: WorkerManager | None = None


def get_worker_manager(paths: RuntimePaths | None = None) -> WorkerManager:
    """Return the global singleton WorkerManager."""
    global _worker_manager
    if _worker_manager is None:
        if paths is None:
            paths = RuntimePaths.from_config()
        _worker_manager = WorkerManager(paths)
    return _worker_manager
