"""Canvas Query Engine — query InteractionCanvas for candidates matching a target."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.perception.page_compiler_models import Candidate, InteractionCanvas, SemanticRole


@dataclass
class QueryTarget:
    """Query target specification. Exactly one field should be set."""
    text: str | None = None                    # Fuzzy text match
    semantic_role: str | None = None           # Exact semantic role match
    region: str | None = None                  # Region ID or role
    natural_language: str | None = None        # Keyword extraction + combo
    composite: dict[str, Any] | None = None    # Arbitrary filter dict


@dataclass
class QueryResult:
    """Result of a canvas query."""
    candidates: list[Candidate] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    total_matched: int = 0
    query_target: QueryTarget | None = None


class CanvasQueryEngine:
    """Query an InteractionCanvas for candidates matching a target."""

    def query(
        self,
        canvas: InteractionCanvas,
        target: QueryTarget,
        max_results: int = 5,
        min_confidence: float = 0.2,
    ) -> QueryResult:
        """Query the canvas for candidates matching the target."""
        scored: list[tuple[float, Candidate]] = []

        for candidate in canvas.elements:
            score = self._score_candidate(candidate, target, canvas)
            if score >= min_confidence:
                scored.append((score, candidate))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:max_results]]

        suggestions: list[str] = []
        if not top:
            suggestions = self._build_suggestions(canvas, target)

        return QueryResult(
            candidates=top,
            suggestions=suggestions,
            total_matched=len(scored),
            query_target=target,
        )

    def _score_candidate(
        self,
        candidate: Candidate,
        target: QueryTarget,
        canvas: InteractionCanvas,
    ) -> float:
        """Score a candidate against the query target. Returns 0-1."""
        scores: list[float] = []

        if target.text is not None:
            scores.append(self._text_score(candidate, target.text))

        if target.semantic_role is not None:
            role_val = candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role)
            scores.append(1.0 if role_val == target.semantic_role else 0.0)

        if target.region is not None:
            in_region = (candidate.region_id == target.region)
            # Also check region role
            region = canvas.get_region(candidate.region_id) if candidate.region_id else None
            region_match = region is not None and region.role == target.region
            scores.append(1.0 if (in_region or region_match) else 0.0)

        if target.natural_language is not None:
            scores.append(self._nl_score(candidate, target.natural_language))

        if target.composite is not None:
            scores.append(self._composite_score(candidate, target.composite))

        if not scores:
            return 0.0

        # Relevance: average of all matched dimensions
        relevance_score = sum(scores) / len(scores)

        # Phase 7: reliability weight from fused_confidence
        # Maps fused_confidence [0, 1] → reliability_weight [0.5, 1.0]
        raw_confidence = getattr(candidate, "confidence", 0.0)
        try:
            fused = float(raw_confidence if raw_confidence is not None else 0.0)
        except (TypeError, ValueError):
            fused = 0.0
        fused = max(0.0, min(1.0, fused))
        reliability_weight = 0.5 + 0.5 * fused

        return relevance_score * reliability_weight

    def _text_score(self, candidate: Candidate, query: str) -> float:
        """Fuzzy text match score."""
        query_lower = query.lower()
        texts = [
            candidate.text or "",
            candidate.name or "",
            candidate.placeholder or "",
            candidate.role_label or "",  # refined role_label
        ]
        best = 0.0
        for t in texts:
            t_lower = t.lower()
            if query_lower in t_lower:
                # Exact substring match
                ratio = len(query_lower) / max(len(t_lower), 1)
                best = max(best, 0.5 + 0.5 * ratio)
            else:
                # Fuzzy match
                ratio = SequenceMatcher(None, query_lower, t_lower).ratio()
                best = max(best, ratio * 0.7)
        return best

    def _nl_score(self, candidate: Candidate, query: str) -> float:
        """Natural language query score — keyword extraction + combo."""
        keywords = query.lower().split()
        if not keywords:
            return 0.0

        texts = " ".join([
            candidate.text or "",
            candidate.name or "",
            candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role),
            candidate.control_type or "",
            candidate.role_label or "",           # refined role_label
            " ".join(candidate.semantic_tags),    # refined semantic_tags
            candidate.visual_type or "",          # refined visual_type
        ]).lower()

        matched = sum(1 for kw in keywords if kw in texts)
        return matched / len(keywords)

    def _composite_score(self, candidate: Candidate, filters: dict[str, Any]) -> float:
        """Composite filter score."""
        matches = 0
        total = 0

        if "text" in filters:
            total += 1
            if self._text_score(candidate, filters["text"]) > 0.3:
                matches += 1

        if "semantic_role" in filters:
            total += 1
            role_val = candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role)
            if role_val == filters["semantic_role"]:
                matches += 1

        if "region" in filters:
            total += 1
            if candidate.region_id == filters["region"]:
                matches += 1

        if "interactable" in filters:
            total += 1
            if candidate.interactable == filters["interactable"]:
                matches += 1

        if "control_type" in filters:
            total += 1
            if candidate.control_type == filters["control_type"]:
                matches += 1

        # Refine field filters
        if "role_label" in filters:
            total += 1
            if candidate.role_label == filters["role_label"]:
                matches += 1

        if "semantic_tags" in filters:
            total += 1
            tag_filter = filters["semantic_tags"]
            if isinstance(tag_filter, str):
                tag_filter = [tag_filter]
            if any(t in candidate.semantic_tags for t in tag_filter):
                matches += 1

        if "visual_type" in filters:
            total += 1
            if candidate.visual_type == filters["visual_type"]:
                matches += 1

        if "refine_status" in filters:
            total += 1
            if candidate.refine_status == filters["refine_status"]:
                matches += 1

        return matches / max(total, 1)

    def _build_suggestions(
        self,
        canvas: InteractionCanvas,
        target: QueryTarget,
    ) -> list[str]:
        """Build suggestions when no candidates found."""
        suggestions: list[str] = []

        if target.text:
            suggestions.append(f"No element with text '{target.text}' found. Try scrolling the page or using allow_vlm=true.")

        if target.semantic_role:
            suggestions.append(f"No element with role '{target.semantic_role}' found. The page may need re-observation.")

        if canvas.partial:
            suggestions.append("Canvas is partial (some providers failed). Results may be incomplete.")

        if not canvas.stable:
            suggestions.append("Canvas is not stable (page may be loading). Wait and re-observe.")

        return suggestions
