"""Check OpenClaw runtime ports before starting local services.

This guard is intentionally read-only: it probes configured endpoints and
reports conflicts, but it does not start or stop any process.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RuntimeTargets:
    api_base_url: str
    api_health_url: str
    vision_endpoint: str
    vision_probe_url: str
    vision_autostart: bool


@dataclass(frozen=True)
class HttpProbe:
    url: str
    ok: bool
    status: int | None = None
    error: str = ""
    body_excerpt: str = ""


@dataclass(frozen=True)
class RuntimeIssue:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class RuntimePortReport:
    ok: bool
    targets: RuntimeTargets
    probes: dict[str, HttpProbe]
    issues: list[RuntimeIssue]


HttpGet = Callable[[str], HttpProbe]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _normalize_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}".replace("localhost", "127.0.0.1").rstrip("/")


def _normalize_parse_endpoint(endpoint: str) -> str:
    value = str(endpoint or "").strip() or "http://127.0.0.1:8001/parse/"
    if value.endswith("/probe") or value.endswith("/probe/"):
        value = value.rstrip("/")[:-5] + "/parse/"
    elif not value.rstrip("/").endswith("/parse"):
        value = value.rstrip("/") + "/parse/"
    elif not value.endswith("/"):
        value += "/"
    return value.replace("localhost", "127.0.0.1")


def _normalize_probe_endpoint(endpoint: str, parse_endpoint: str) -> str:
    value = str(endpoint or "").strip()
    if not value:
        value = _normalize_parse_endpoint(parse_endpoint).replace("/parse/", "/probe/")
    if value.rstrip("/").endswith("/parse"):
        value = value.rstrip("/")[:-5] + "/probe/"
    elif not value.rstrip("/").endswith("/probe"):
        value = value.rstrip("/") + "/probe/"
    elif not value.endswith("/"):
        value += "/"
    return value.replace("localhost", "127.0.0.1")


def _origin(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    default_port = 443 if scheme == "https" else 80
    return (parsed.hostname or "127.0.0.1", int(parsed.port or default_port))


def load_targets(root: Path = ROOT) -> RuntimeTargets:
    app_yaml = _load_yaml(root / "config" / "app.yaml")
    models_yaml = _load_yaml(root / "config" / "models.yaml")
    api_section = app_yaml.get("api", {})
    models_root = models_yaml.get("models", models_yaml)
    vision_section = models_root.get("vision", {})

    api_base_url = _normalize_base_url(
        str(api_section.get("host", "127.0.0.1")),
        int(api_section.get("port", 8000)),
    )
    vision_endpoint = _normalize_parse_endpoint(
        str(vision_section.get("endpoint", "http://127.0.0.1:8001/parse/"))
    )
    vision_probe_url = _normalize_probe_endpoint(
        str(vision_section.get("probe_endpoint", "")),
        vision_endpoint,
    )
    return RuntimeTargets(
        api_base_url=api_base_url,
        api_health_url=f"{api_base_url}/health",
        vision_endpoint=vision_endpoint,
        vision_probe_url=vision_probe_url,
        vision_autostart=bool(vision_section.get("autostart", True)),
    )


def http_get(url: str, timeout: float = 2.0) -> HttpProbe:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 0.75)):
            pass
    except (ConnectionRefusedError, TimeoutError, socket.timeout, OSError) as exc:
        return HttpProbe(url=url, ok=False, error=type(exc).__name__, body_excerpt=str(exc)[:200])

    request = Request(url, headers={"User-Agent": "openclaw-runtime-port-check/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            return HttpProbe(url=url, ok=200 <= status < 300, status=status, body_excerpt=body[:200])
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return HttpProbe(
            url=url,
            ok=False,
            status=int(exc.code),
            error=f"http_{exc.code}",
            body_excerpt=body[:200],
        )
    except (ConnectionRefusedError, TimeoutError, socket.timeout, URLError, OSError) as exc:
        return HttpProbe(url=url, ok=False, error=type(exc).__name__, body_excerpt=str(exc)[:200])


def _healthy_openclaw(probe: HttpProbe) -> bool:
    if not probe.ok:
        return False
    body = probe.body_excerpt.lower()
    return "healthy" in body or '"status"' in body


def evaluate_runtime_targets(
    targets: RuntimeTargets,
    get: HttpGet = http_get,
) -> RuntimePortReport:
    probes: dict[str, HttpProbe] = {}
    issues: list[RuntimeIssue] = []

    if _origin(targets.api_base_url) == _origin(targets.vision_endpoint):
        issues.append(
            RuntimeIssue(
                level="error",
                code="api_vision_same_port",
                message=(
                    "API and OmniParser are configured on the same host/port: "
                    f"{targets.api_base_url} vs {targets.vision_endpoint}"
                ),
            )
        )

    api_probe = get(targets.api_health_url)
    probes["api_health"] = api_probe
    if api_probe.ok and not _healthy_openclaw(api_probe):
        issues.append(
            RuntimeIssue(
                level="warning",
                code="api_health_unexpected_body",
                message=f"API /health responded but did not look like OpenClaw: {api_probe.body_excerpt[:80]}",
            )
        )
    elif api_probe.status is not None and not api_probe.ok:
        issues.append(
            RuntimeIssue(
                level="error",
                code="api_port_occupied_or_unhealthy",
                message=f"API health endpoint returned HTTP {api_probe.status}: {targets.api_health_url}",
            )
        )

    vision_probe = get(targets.vision_probe_url)
    probes["vision_probe"] = vision_probe
    if vision_probe.ok:
        pass
    elif vision_probe.status is not None:
        vision_host, vision_port = _origin(targets.vision_probe_url)
        vision_scheme = urlparse(targets.vision_probe_url).scheme or "http"
        vision_health_url = f"{vision_scheme}://{vision_host}:{vision_port}/health"
        vision_health_probe = get(vision_health_url)
        probes["vision_port_health"] = vision_health_probe
        if _healthy_openclaw(vision_health_probe):
            issues.append(
                RuntimeIssue(
                    level="error",
                    code="openclaw_api_on_vision_port",
                    message=(
                        "Vision probe failed and the same port responds like OpenClaw API; "
                        f"free the OmniParser port before observe: {targets.vision_probe_url}"
                    ),
                )
            )
        else:
            issues.append(
                RuntimeIssue(
                    level="error",
                    code="vision_probe_http_error",
                    message=f"Vision probe returned HTTP {vision_probe.status}: {targets.vision_probe_url}",
                )
            )
    elif not targets.vision_autostart:
        issues.append(
            RuntimeIssue(
                level="warning",
                code="vision_not_running_autostart_disabled",
                message=f"Vision provider is not reachable and autostart is disabled: {targets.vision_probe_url}",
            )
        )

    ok = not any(issue.level == "error" for issue in issues)
    return RuntimePortReport(ok=ok, targets=targets, probes=probes, issues=issues)


def print_text_report(report: RuntimePortReport) -> None:
    print("OpenClaw runtime port check")
    print(f"- API: {report.targets.api_health_url}")
    print(f"- Vision parse: {report.targets.vision_endpoint}")
    print(f"- Vision probe: {report.targets.vision_probe_url}")
    for name, probe in report.probes.items():
        status = probe.status if probe.status is not None else probe.error or "not_reachable"
        print(f"- {name}: {'ok' if probe.ok else 'fail'} ({status})")
    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"- [{issue.level}] {issue.code}: {issue.message}")
    else:
        print("No blocking runtime port issues detected.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check OpenClaw API and OmniParser port health.")
    parser.add_argument("--root", default=str(ROOT), help="Project root containing config/app.yaml and config/models.yaml")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    report = evaluate_runtime_targets(load_targets(Path(args.root)))
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print_text_report(report)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
