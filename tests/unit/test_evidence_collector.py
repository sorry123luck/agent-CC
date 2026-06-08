from src.memory.evidence_collector import _normalize_provider


def test_normalize_provider_maps_local_layout_sources_to_observe_provider():
    assert _normalize_provider("app_layout") == "uia"
    assert _normalize_provider("boundary_candidate") == "uia"
