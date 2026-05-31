"""In-memory canvas cache for the core protocol loop.

Stores InteractionCanvas instances keyed by canvas_id.
Query, diff, remember, and feedback all retrieve canvases from here.

Design: process-local OrderedDict with LRU eviction (L1) plus optional
disk backed warm cache (L2) for screenshot persistence across restarts,
and DB persistence (L3) for full canvas serialization across restarts.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from PIL import Image

from src.perception.page_compiler_models import InteractionCanvas

logger = logging.getLogger(__name__)

_MAX_CANVASES = 64

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_WARM_DIR = _ROOT_DIR / "data" / "visual_assets" / "warm"


def _safe_filename(canvas_id: str) -> str:
    """Sanitize canvas_id for use as a filename (defense-in-depth)."""
    return re.sub(r"[^\w\-.]", "_", canvas_id)


class CanvasCache:
    """Thread-safe LRU cache for InteractionCanvas instances with multi-tier persistence.

    L1 (memory): OrderedDict-based LRU, configurable max_size.
    L2 (disk): JPEG warm cache at data/visual_assets/warm/{canvas_id}.jpg.
       Written in a background daemon thread when persist=True is passed to put().
       FIFO eviction by file mtime when file count exceeds l2_max_files.
    L3 (DB): Full canvas JSON stored in canvas_snapshots.canvas_json.
       Written in the same daemon thread as L2. Restored on L1 miss.
    """

    def __init__(
        self,
        max_size: int = _MAX_CANVASES,
        l2_max_files: int = 300,
        l2_jpeg_quality: int = 80,
    ) -> None:
        self._cache: OrderedDict[str, InteractionCanvas] = OrderedDict()
        self._screenshots: OrderedDict[str, Image.Image] = OrderedDict()
        self._deleted_canvas_ids: set[str] = set()
        self._max_size = max_size
        self._l2_max_files = l2_max_files
        self._l2_jpeg_quality = l2_jpeg_quality
        self._lock = Lock()

    def put(
        self,
        canvas: InteractionCanvas,
        screenshot: Image.Image | None = None,
        persist: bool = False,
    ) -> None:
        """Store a canvas (and optional screenshot). Evicts oldest if at capacity.

        Args:
            persist: If True, write screenshot to L2 disk and canvas JSON to L3 DB
                     in a background thread after storing in memory.
        """
        with self._lock:
            if canvas.canvas_id in self._deleted_canvas_ids:
                logger.debug("Ignoring put for deleted canvas %s", canvas.canvas_id)
                return
            if canvas.canvas_id in self._cache:
                self._cache.move_to_end(canvas.canvas_id)
            self._cache[canvas.canvas_id] = canvas
            if screenshot is not None:
                self._screenshots[canvas.canvas_id] = screenshot
            while len(self._cache) > self._max_size:
                evicted_id, _ = self._cache.popitem(last=False)
                self._screenshots.pop(evicted_id, None)
                logger.debug("Evicted canvas %s from cache", evicted_id)

        # L2/L3 write happens outside the lock to avoid blocking reads
        if persist:
            self._persist_to_l2_and_db(canvas, screenshot)

    def get(self, canvas_id: str) -> InteractionCanvas | None:
        """Retrieve a canvas by ID. Falls back to DB on L1 miss. Returns None if not found."""
        with self._lock:
            if canvas_id in self._deleted_canvas_ids:
                return None
            canvas = self._cache.get(canvas_id)
            if canvas is not None:
                self._cache.move_to_end(canvas_id)
                return canvas

        # L1 miss — try DB (outside lock to avoid holding it during I/O)
        canvas = CanvasCache._restore_from_db(canvas_id)
        if canvas is not None:
            with self._lock:
                self._cache[canvas_id] = canvas
                while len(self._cache) > self._max_size:
                    evicted_id, _ = self._cache.popitem(last=False)
                    self._screenshots.pop(evicted_id, None)
            logger.debug("Restored canvas %s from DB", canvas_id)
        return canvas

    def get_screenshot(self, canvas_id: str) -> Image.Image | None:
        """Retrieve a screenshot by canvas ID. Falls back to L2 disk cache."""
        with self._lock:
            if canvas_id in self._deleted_canvas_ids:
                return None
            img = self._screenshots.get(canvas_id)
            if img is not None:
                return img
        # L1 miss — try L2 disk (outside lock to avoid holding it during I/O)
        img = CanvasCache.get_screenshot_from_disk(canvas_id)
        if img is not None:
            with self._lock:
                self._screenshots[canvas_id] = img
        return img

    def put_screenshot(self, canvas_id: str, screenshot: Image.Image) -> None:
        """Store a screenshot for an existing canvas."""
        with self._lock:
            if canvas_id in self._cache:
                self._screenshots[canvas_id] = screenshot

    def list_all(self) -> list[tuple[str, InteractionCanvas]]:
        """Return all cached (canvas_id, canvas) pairs without changing LRU order."""
        with self._lock:
            return [
                (canvas_id, canvas)
                for canvas_id, canvas in self._cache.items()
                if canvas_id not in self._deleted_canvas_ids
            ]

    def is_deleted(self, canvas_id: str) -> bool:
        """Return whether a canvas id was explicitly deleted in this process."""
        with self._lock:
            return canvas_id in self._deleted_canvas_ids

    def remove(self, canvas_id: str) -> bool:
        """Remove a canvas (and its screenshot) from cache. Returns True if found."""
        with self._lock:
            self._deleted_canvas_ids.add(canvas_id)
            existed = self._cache.pop(canvas_id, None) is not None
            screenshot_existed = self._screenshots.pop(canvas_id, None) is not None
            return existed or screenshot_existed

    def remove_persistent(self, canvas_id: str) -> dict[str, bool]:
        """Remove a canvas from L1 memory and L2 warm screenshot storage."""
        memory_removed = self.remove(canvas_id)
        warm_removed = False
        path = _WARM_DIR / f"{_safe_filename(canvas_id)}.jpg"
        try:
            if path.exists():
                path.unlink()
                warm_removed = True
        except OSError:
            logger.warning("L2 warm delete failed for canvas %s", canvas_id, exc_info=True)
        return {
            "memory": memory_removed,
            "warm": warm_removed,
        }

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._screenshots.clear()
            self._deleted_canvas_ids.clear()

    # ------------------------------------------------------------------
    # L2 warm cache
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_warm_dir() -> Path:
        """Create the warm cache directory if it doesn't exist."""
        _WARM_DIR.mkdir(parents=True, exist_ok=True)
        return _WARM_DIR

    def _write_l2_warm(self, canvas_id: str, screenshot: Image.Image) -> None:
        """Write screenshot to L2 warm cache. Runs in a daemon thread."""
        try:
            warm_dir = self._ensure_warm_dir()
            path = warm_dir / f"{_safe_filename(canvas_id)}.jpg"
            img = screenshot if screenshot.mode == "RGB" else screenshot.convert("RGB")
            img.save(str(path), format="JPEG", quality=self._l2_jpeg_quality)
            logger.debug("L2 warm write: %s (%d bytes)", path.name, path.stat().st_size)
        except Exception:
            logger.warning("L2 warm write failed for canvas %s", canvas_id, exc_info=True)

    def _evict_l2_fifo(self) -> None:
        """Evict oldest files from L2 warm cache if over capacity."""
        try:
            warm_dir = self._ensure_warm_dir()
            files = sorted(warm_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            excess = len(files) - self._l2_max_files
            if excess <= 0:
                return
            for f in files[:excess]:
                try:
                    f.unlink()
                    logger.debug("L2 FIFO evicted: %s", f.name)
                except OSError:
                    logger.warning("L2 eviction failed for %s", f.name, exc_info=True)
        except Exception:
            logger.warning("L2 eviction scan failed", exc_info=True)

    def _persist_to_l2_and_db(
        self, canvas: InteractionCanvas, screenshot: Image.Image | None
    ) -> None:
        """Persist screenshot to L2 and canvas JSON to DB in a background thread.

        Non-blocking: failures are logged as warnings, never affect the caller.
        """
        def _worker() -> None:
            if self.is_deleted(canvas.canvas_id):
                return
            if screenshot is not None:
                self._write_l2_warm(canvas.canvas_id, screenshot)
                if self.is_deleted(canvas.canvas_id):
                    path = _WARM_DIR / f"{_safe_filename(canvas.canvas_id)}.jpg"
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Late L2 warm delete failed for canvas %s", canvas.canvas_id, exc_info=True)
                    return
                self._evict_l2_fifo()
            if self.is_deleted(canvas.canvas_id):
                return
            try:
                self._write_canvas_json_to_db(canvas, self.is_deleted)
            except Exception:
                logger.warning(
                    "L3 canvas JSON write failed for canvas %s", canvas.canvas_id, exc_info=True,
                )

        canvas_id = canvas.canvas_id
        t = threading.Thread(
            target=_worker, daemon=True, name=f"l2-write-{canvas_id[:8]}",
        )
        t.start()

    @staticmethod
    def _write_canvas_json_to_db(canvas: InteractionCanvas, is_deleted=None) -> None:
        """Serialize canvas to JSON and upsert into canvas_snapshots.canvas_json col."""
        if is_deleted is not None and is_deleted(canvas.canvas_id):
            return
        from src.canvas.canvas_serializer import dumps as canvas_dumps
        json_str = canvas_dumps(canvas)

        from src.storage.db import Session
        from src.storage.schema import CanvasSnapshotRecord
        from sqlalchemy import text

        with Session() as session:
            if is_deleted is not None and is_deleted(canvas.canvas_id):
                return
            row = session.query(CanvasSnapshotRecord).filter(
                CanvasSnapshotRecord.canvas_id == canvas.canvas_id,
            ).first()
            if row is not None:
                row.canvas_json = json_str
                row.element_count = len(canvas.elements)
                row.has_screenshot = 1
                row.page_model_id = getattr(canvas, "page_model_id", None) or row.page_model_id
                row.state_template_id = getattr(canvas, "state_template_id", None) or row.state_template_id
            else:
                # INSERT fallback: record_canvas_snapshot() should have created the row,
                # but if the background thread doesn't see it yet, include page_model_id
                # and state_template_id from the canvas object itself.
                from datetime import datetime, timezone
                import uuid
                session.execute(
                    text(
                        "INSERT INTO canvas_snapshots "
                        "(snapshot_id, canvas_id, captured_at, element_count, has_screenshot, "
                        "canvas_json, page_model_id, state_template_id) "
                        "VALUES (:sid, :cid, :cap, :ec, :hs, :cj, :pm, :st)"
                        " ON CONFLICT(canvas_id) DO UPDATE SET "
                        "element_count = excluded.element_count, "
                        "has_screenshot = excluded.has_screenshot, "
                        "canvas_json = excluded.canvas_json, "
                        "page_model_id = COALESCE(excluded.page_model_id, canvas_snapshots.page_model_id), "
                        "state_template_id = COALESCE(excluded.state_template_id, canvas_snapshots.state_template_id)"
                    ),
                    {
                        "sid": str(uuid.uuid4()),
                        "cid": canvas.canvas_id,
                        "cap": datetime.now(timezone.utc).isoformat(),
                        "ec": len(canvas.elements),
                        "hs": 1,
                        "cj": json_str,
                        "pm": getattr(canvas, "page_model_id", None),
                        "st": getattr(canvas, "state_template_id", None),
                    },
                )
            session.commit()

    @staticmethod
    def _restore_from_db(canvas_id: str) -> InteractionCanvas | None:
        """Load a canvas from canvas_snapshots.canvas_json. Returns None if not found."""
        try:
            from src.storage.db import Session
            from src.storage.schema import CanvasSnapshotRecord

            with Session() as session:
                row = session.query(CanvasSnapshotRecord).filter(
                    CanvasSnapshotRecord.canvas_id == canvas_id,
                ).first()
                if row is None or not row.canvas_json:
                    return None

                from src.canvas.canvas_serializer import loads as canvas_loads
                return canvas_loads(row.canvas_json)
        except Exception:
            logger.warning(
                "DB restore failed for canvas %s", canvas_id, exc_info=True,
            )
            return None

    @staticmethod
    def get_screenshot_from_disk(canvas_id: str) -> Image.Image | None:
        """Load a screenshot from L2 warm cache. Returns None if not found."""
        path = _WARM_DIR / f"{_safe_filename(canvas_id)}.jpg"
        if not path.exists():
            return None
        try:
            return Image.open(str(path)).convert("RGB")
        except Exception:
            logger.warning("L2 warm read failed for canvas %s", canvas_id, exc_info=True)
            return None


# --- Lazy singleton (reads config on first access) ---
# Double-checked locking is safe under CPython's GIL. If porting to a
# non-GIL runtime (PyPy, GraalPy, free-threaded CPython 3.13+), replace
# with a threading-safe lazy init pattern.
_singleton: CanvasCache | None = None
_singleton_lock = Lock()


def get_canvas_cache() -> CanvasCache:
    """Return the process-global CanvasCache singleton, initialized from config on first call."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        from src.common.config_manager import load_config
        cfg = load_config().cache
        _singleton = CanvasCache(
            max_size=cfg.l1_max_size,
            l2_max_files=cfg.l2_max_files,
            l2_jpeg_quality=cfg.l2_jpeg_quality,
        )
        return _singleton
