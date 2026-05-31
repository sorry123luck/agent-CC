from scripts.check_runtime_ports import (
    HttpProbe,
    RuntimeTargets,
    evaluate_runtime_targets,
    load_targets,
)


def _targets() -> RuntimeTargets:
    return RuntimeTargets(
        api_base_url="http://127.0.0.1:8000",
        api_health_url="http://127.0.0.1:8000/health",
        vision_endpoint="http://127.0.0.1:8001/parse/",
        vision_probe_url="http://127.0.0.1:8001/probe/",
        vision_autostart=True,
    )


def test_runtime_guard_allows_clean_pre_start_state():
    def get(url: str) -> HttpProbe:
        return HttpProbe(url=url, ok=False, error="ConnectionRefusedError")

    report = evaluate_runtime_targets(_targets(), get=get)

    assert report.ok is True
    assert report.issues == []
    assert report.probes["api_health"].error == "ConnectionRefusedError"


def test_runtime_guard_detects_openclaw_api_on_vision_port():
    def get(url: str) -> HttpProbe:
        if url.endswith("/health"):
            return HttpProbe(url=url, ok=True, status=200, body_excerpt='{"status":"healthy"}')
        return HttpProbe(url=url, ok=False, status=404, error="http_404", body_excerpt='{"detail":"Not Found"}')

    report = evaluate_runtime_targets(_targets(), get=get)

    assert report.ok is False
    assert any(issue.code == "openclaw_api_on_vision_port" for issue in report.issues)
    assert report.probes["vision_probe"].status == 404


def test_runtime_guard_detects_configured_port_collision():
    targets = RuntimeTargets(
        api_base_url="http://127.0.0.1:8001",
        api_health_url="http://127.0.0.1:8001/health",
        vision_endpoint="http://127.0.0.1:8001/parse/",
        vision_probe_url="http://127.0.0.1:8001/probe/",
        vision_autostart=True,
    )

    def get(url: str) -> HttpProbe:
        return HttpProbe(url=url, ok=False, error="ConnectionRefusedError")

    report = evaluate_runtime_targets(targets, get=get)

    assert report.ok is False
    assert any(issue.code == "api_vision_same_port" for issue in report.issues)


def test_runtime_guard_loads_current_config():
    targets = load_targets()

    assert targets.api_health_url == "http://127.0.0.1:8000/health"
    assert targets.vision_endpoint == "http://127.0.0.1:8001/parse/"
    assert targets.vision_probe_url == "http://127.0.0.1:8001/probe/"
