# OpenClaw Desktop Agent Architecture

OpenClaw is a Windows desktop perception and action-planning substrate for AI agents. It turns real application windows into a structured `InteractionCanvas`, then exposes query, diff, memory, readback, and guarded action APIs so an external agent can decide what to do.

The project is intentionally not a chat bot and does not hard-code private app workflows. The local service focuses on:

- observing native desktop windows with UIA, OCR, geometry, and optional vision-model supplements;
- producing candidates with coordinates, roles, confidence, evidence, and risk metadata;
- preserving state transitions and page memories for later agent use;
- providing a guarded `/api/v1/act` adapter with explicit policy classes;
- keeping real screenshots, runtime databases, and user artifacts outside the public repository.

## Current Status

The codebase is in an active prototype stage. The core architecture, console UI, perception pipeline, VLM supplement layer, page operability reports, action preflight, controlled click/scroll, readback verification, and regression scripts are present. Live private sample artifacts are intentionally excluded from this public branch.

Public progress summaries are maintained in:

- [STATUS.md](STATUS.md)
- [CURRENT_MILESTONE.md](CURRENT_MILESTONE.md)
- [docs/public/progress-summary.md](docs/public/progress-summary.md)
- [docs/changelog/2026-05-public-progress.md](docs/changelog/2026-05-public-progress.md)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/integration` | FastAPI service and HTTP models |
| `src/perception` | UIA/OCR/geometry/VLM fusion and page compiler |
| `src/canvas` | canvas cache, query, diff, refinement |
| `src/execution` | guarded action service, policy, verification |
| `src/memory` | page templates, evidence, transitions, feedback |
| `src/windows` | window enumeration, screenshot, activation context |
| `src/vlm` | ROI contracts, provider workers, prompt helpers |
| `src/chat` | generic message stream crop/readback helpers used by tests |
| `src/ui/console` | Vite/React inspection console |
| `scripts` | regression, readiness, analysis, and audit tooling |
| `tests` | unit, e2e, and real-app test harnesses |

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.integration.api_server
```

In a second terminal:

```powershell
cd src\ui\console
npm install
npm run dev
```

Then open the console at the Vite URL, usually `http://127.0.0.1:5173/`.

## Configuration

The public repo does not include local model paths, API keys, screenshots, or runtime databases. Start from:

- [config/examples/models.example.yaml](config/examples/models.example.yaml)

Copy it to a local ignored config path and fill in providers through environment variables. Keep secrets out of git.

## Safety Model

OpenClaw separates perception from decision making:

- `read-only`: observe, query, diff, read regions, and report evidence.
- `review`: candidate is visible but needs caller review.
- `controlled`: bounded click/scroll actions with verification.
- `blocked`: typing, sending, destructive, or ambiguous actions unless a stricter gate approves them.

Text input and send-like actions remain intentionally conservative. The service should provide enough structure for an agent to make a plan, but it should not silently infer user intent or send messages on its own.

## Tests

Run a focused public smoke suite:

```powershell
python -m pytest tests\unit\test_analyze_goal_progress.py tests\unit\test_analyze_sample_coverage.py tests\unit\test_run_agent_operability_regression.py tests\unit\test_risk_policy.py -q
```

Full real-app tests require a Windows desktop session and local software state. They are kept as harnesses, not as CI requirements.

## Privacy

Generated files under `artifacts/`, runtime databases under `data/`, screenshots, logs, and local configs are ignored. Public docs summarize results without including private screenshots or chat/window evidence.

