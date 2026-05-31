"""Canvas Diff Engine — compare two InteractionCanvas snapshots and report changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from src.perception.page_compiler_models import Candidate, InteractionCanvas

if TYPE_CHECKING:
    from src.memory.candidate_identity import CandidateSignature
    from src.memory.candidate_key_store import CandidateKeyStore

SIGNATURE_MATCH_THRESHOLD = 0.75


@dataclass
class CanvasChanges:
    """Changes between two InteractionCanvas snapshots."""
    added: list[Candidate] = field(default_factory=list)
    removed: list[Candidate] = field(default_factory=list)
    preserved: list[Candidate] = field(default_factory=list)
    page_changed: bool = False
    page_class_changed: bool = False
    summary: str = ""


class CanvasDiffEngine:
    """Compare two InteractionCanvas snapshots and produce a change report.

    匹配策略链：
    1. stable_key_id 精确匹配（如果两个候选都有 key）
    2. compute_signature_similarity >= 0.75（需要 key_store + canvas）
    3. 同 role + IoU > 0.5
    4. 同 role + text > 0.8
    """

    def __init__(
        self,
        key_store: CandidateKeyStore | None = None,
        before_canvas: InteractionCanvas | None = None,
        after_canvas: InteractionCanvas | None = None,
    ) -> None:
        self._key_store = key_store
        self._before_canvas = before_canvas
        self._after_canvas = after_canvas
        self._signature_cache: dict[str, CandidateSignature] = {}

    def diff(
        self,
        before: InteractionCanvas,
        after: InteractionCanvas,
        detail_level: str = "changes",
    ) -> CanvasChanges:
        """Compare two canvases and return changes."""
        matched_after_ids: set[str] = set()
        matched_before_ids: set[str] = set()
        preserved: list[Candidate] = []

        for after_elem in after.elements:
            best_match = self._find_match(after_elem, before.elements, before, after)
            if best_match is not None:
                preserved.append(after_elem)
                matched_after_ids.add(after_elem.element_id)
                matched_before_ids.add(best_match.element_id)

        added = [e for e in after.elements if e.element_id not in matched_after_ids]
        removed = [e for e in before.elements if e.element_id not in matched_before_ids]

        page_changed = before.page_class != after.page_class
        page_class_changed = before.page.page_class != after.page.page_class

        summary = self._build_summary(added, removed, preserved, page_changed)

        return CanvasChanges(
            added=added,
            removed=removed,
            preserved=preserved,
            page_changed=page_changed,
            page_class_changed=page_class_changed,
            summary=summary,
        )

    def _find_match(
        self,
        target: Candidate,
        candidates: list[Candidate],
        before: InteractionCanvas | None = None,
        after: InteractionCanvas | None = None,
    ) -> Candidate | None:
        """Find the best matching candidate in the list.

        Strategy chain:
        1. stable_key_id exact match
        2. signature similarity >= 0.75
        3. same role + IoU > 0.5
        4. same role + text > 0.8
        """
        # Strategy 1: stable_key_id exact match
        if target.stable_key_id:
            for c in candidates:
                if c.stable_key_id == target.stable_key_id:
                    return c

        # Strategy 2: signature similarity
        if self._key_store and before and after:
            match = self._signature_match(target, candidates, before, after)
            if match is not None:
                return match

        # Strategy 3: same role + similar bounds
        target_role = self._role_str(target)
        for c in candidates:
            if self._role_str(c) == target_role and target.bounds and c.bounds:
                iou = self._bounds_iou(target.bounds, c.bounds)
                if iou > 0.5:
                    return c

        # Strategy 4: same role + similar text
        for c in candidates:
            if self._role_str(c) == target_role:
                t_text = (target.text or "").strip()
                c_text = (c.text or "").strip()
                if t_text and c_text:
                    ratio = SequenceMatcher(None, t_text, c_text).ratio()
                    if ratio > 0.8:
                        return c

        return None

    def _signature_match(
        self,
        target: Candidate,
        candidates: list[Candidate],
        before: InteractionCanvas,
        after: InteractionCanvas,
    ) -> Candidate | None:
        """Signature-based matching."""
        from src.memory.candidate_identity import build_signature, compute_signature_similarity

        target_sig = self._get_signature(target, after)
        if target_sig is None:
            return None

        best_match = None
        best_sim = 0.0

        for c in candidates:
            c_sig = self._get_signature(c, before)
            if c_sig is None:
                continue
            sim = compute_signature_similarity(target_sig, c_sig)
            if sim > best_sim:
                best_sim = sim
                best_match = c

        if best_match and best_sim >= SIGNATURE_MATCH_THRESHOLD:
            return best_match
        return None

    def _get_signature(
        self,
        candidate: Candidate,
        canvas: InteractionCanvas,
    ) -> CandidateSignature | None:
        """Get or compute signature (cached)."""
        cache_key = f"{canvas.canvas_id}:{candidate.element_id}"
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]

        from src.memory.candidate_identity import build_signature
        try:
            sig = build_signature(candidate, canvas)
            self._signature_cache[cache_key] = sig
            return sig
        except Exception:
            return None

    def _role_str(self, candidate: Candidate) -> str:
        """Get semantic role as string."""
        r = candidate.semantic_role
        return r.value if hasattr(r, "value") else str(r)

    def _bounds_iou(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU of two bounding boxes (left, top, right, bottom)."""
        al, at, ar, ab = a
        bl, bt, br, bb = b

        il = max(al, bl)
        it = max(at, bt)
        ir = min(ar, br)
        ib = min(ab, bb)

        if ir <= il or ib <= it:
            return 0.0

        intersection = (ir - il) * (ib - it)
        area_a = (ar - al) * (ab - at)
        area_b = (br - bl) * (bb - bt)
        union = area_a + area_b - intersection

        return intersection / max(union, 1)

    def _build_summary(
        self,
        added: list[Candidate],
        removed: list[Candidate],
        preserved: list[Candidate],
        page_changed: bool,
    ) -> str:
        """Build a human-readable summary of changes."""
        parts: list[str] = []

        if page_changed:
            parts.append("Page changed.")

        if added:
            texts = [e.text or e.name or e.element_id for e in added[:3]]
            parts.append(f"{len(added)} added ({', '.join(texts)}{'...' if len(added) > 3 else ''}).")

        if removed:
            texts = [e.text or e.name or e.element_id for e in removed[:3]]
            parts.append(f"{len(removed)} removed ({', '.join(texts)}{'...' if len(removed) > 3 else ''}).")

        if preserved:
            parts.append(f"{len(preserved)} preserved.")

        if not parts:
            parts.append("No changes detected.")

        return " ".join(parts)
