"""Lazy exports for the perception package."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "SurfaceType",
    "ContentAreaSubtype",
    "SemanticRole",
    "LocatorKind",
    "LocatorStatus",
    "CoordinateSpace",
    "RelationType",
    "AnchorKind",
    "CaptureReason",
    "ConfidenceLevel",
    "RiskTag",
    "RiskLevel",
    "Rect",
    "ElementState",
    "LocatorStats",
    "AppInfo",
    "WindowInfoSnapshot",
    "SurfaceEvidence",
    "SurfaceInfo",
    "PageInfo",
    "ScrollContext",
    "ProviderTrace",
    "Locator",
    "Anchor",
    "Candidate",
    "Region",
    "ElementRelation",
    "InteractionCanvas",
    "PerceptionProvider",
    "SurfaceClassifier",
    "ContentAreaClassifier",
    "ContentAreaClassification",
    "ContentAreaEvidence",
    "PageClassClassifier",
    "PageClassResult",
    "PageClassEvidence",
    "InteractionCanvasEngine",
]

_EXPORT_MAP = {
    "SurfaceType": ("src.perception.page_compiler_models", "SurfaceType"),
    "ContentAreaSubtype": ("src.perception.page_compiler_models", "ContentAreaSubtype"),
    "SemanticRole": ("src.perception.page_compiler_models", "SemanticRole"),
    "LocatorKind": ("src.perception.page_compiler_models", "LocatorKind"),
    "LocatorStatus": ("src.perception.page_compiler_models", "LocatorStatus"),
    "CoordinateSpace": ("src.perception.page_compiler_models", "CoordinateSpace"),
    "RelationType": ("src.perception.page_compiler_models", "RelationType"),
    "AnchorKind": ("src.perception.page_compiler_models", "AnchorKind"),
    "CaptureReason": ("src.perception.page_compiler_models", "CaptureReason"),
    "ConfidenceLevel": ("src.perception.page_compiler_models", "ConfidenceLevel"),
    "RiskTag": ("src.perception.page_compiler_models", "RiskTag"),
    "RiskLevel": ("src.perception.page_compiler_models", "RiskLevel"),
    "Rect": ("src.perception.page_compiler_models", "Rect"),
    "ElementState": ("src.perception.page_compiler_models", "ElementState"),
    "LocatorStats": ("src.perception.page_compiler_models", "LocatorStats"),
    "AppInfo": ("src.perception.page_compiler_models", "AppInfo"),
    "WindowInfoSnapshot": ("src.perception.page_compiler_models", "WindowInfoSnapshot"),
    "SurfaceEvidence": ("src.perception.page_compiler_models", "SurfaceEvidence"),
    "SurfaceInfo": ("src.perception.page_compiler_models", "SurfaceInfo"),
    "PageInfo": ("src.perception.page_compiler_models", "PageInfo"),
    "ScrollContext": ("src.perception.page_compiler_models", "ScrollContext"),
    "ProviderTrace": ("src.perception.page_compiler_models", "ProviderTrace"),
    "Locator": ("src.perception.page_compiler_models", "Locator"),
    "Anchor": ("src.perception.page_compiler_models", "Anchor"),
    "Candidate": ("src.perception.page_compiler_models", "Candidate"),
    "Region": ("src.perception.page_compiler_models", "Region"),
    "ElementRelation": ("src.perception.page_compiler_models", "ElementRelation"),
    "InteractionCanvas": ("src.perception.page_compiler_models", "InteractionCanvas"),
    "PerceptionProvider": ("src.perception.page_compiler_models", "PerceptionProvider"),
    "SurfaceClassifier": ("src.perception.surface_classifier", "SurfaceClassifier"),
    "ContentAreaClassifier": ("src.perception.content_area_classifier", "ContentAreaClassifier"),
    "ContentAreaClassification": ("src.perception.content_area_classifier", "ContentAreaClassification"),
    "ContentAreaEvidence": ("src.perception.content_area_classifier", "ContentAreaEvidence"),
    "PageClassClassifier": ("src.perception.page_class_classifier", "PageClassClassifier"),
    "PageClassResult": ("src.perception.page_class_classifier", "PageClassResult"),
    "PageClassEvidence": ("src.perception.page_class_classifier", "PageClassEvidence"),
    "InteractionCanvasEngine": ("src.perception.page_compiler", "InteractionCanvasEngine"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
