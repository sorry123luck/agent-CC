# OpenClaw

OpenClaw is an experimental Windows desktop perception layer for AI agents.

The goal is to turn a real desktop application window into a structured page model that an external agent can inspect before deciding what to do. The project is still under active development and is not a finished automation product.

## Project Goal

Current desktop agents often rely on raw screenshots and approximate coordinates. OpenClaw explores a middle layer:

1. capture a Windows application window;
2. combine UIA, OCR, geometry, memory, and optional VLM evidence;
3. produce an `InteractionCanvas` with regions, candidate controls, coordinates, labels, confidence, and risk;
4. let an external agent query that model and request guarded actions;
5. verify actions with observe/diff/readback where possible.

The software should provide the map and safety checks. The external agent should still make the task decision.

## Current Status

This repository is a prototype snapshot. The main pieces exist, but the system still needs more real-app testing and cleanup before it can be treated as stable.

Working or partially working:

- Windows window enumeration and screenshot capture.
- `InteractionCanvas` page model.
- UIA/OCR/geometric candidate extraction.
- Optional ROI VLM semantic supplement.
- React inspection console for reviewing canvases and candidates.
- Query, diff, memory, feedback, and action preflight APIs.
- Controlled click/scroll experiments with verification records.
- Regression and analysis scripts for private real-app samples.

Known unfinished areas:

- Input boxes and send-like actions are still conservative and not broadly safe.
- Chat-style apps need more stable composer, message stream, and readback handling.
- VLM results need continued binding/quality checks on more software.
- Multi-page exploration is planned but not finished.
- Public fixtures are minimal because raw screenshots and chat samples are private.
- Setup is source-based; there is no polished installer or packaged release yet.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/integration` | Local API server and request/response models |
| `src/perception` | UIA/OCR/geometry/VLM perception pipeline |
| `src/canvas` | Canvas cache, query, diff, and refinement |
| `src/execution` | Guarded action policy and verification |
| `src/memory` | Page/candidate memory and transition records |
| `src/windows` | Window enumeration, capture, and activation helpers |
| `src/ui/console` | React inspection console |
| `scripts` | Test, analysis, and regression utilities |
| `tests` | Unit/e2e/real-app test harnesses |

## Local Setup

Backend:

```powershell
git clone https://github.com/sorry123luck/agent-CC.git
cd agent-CC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.integration.api_server
```

Frontend console:

```powershell
cd src\ui\console
npm install
npm run dev
```

Then open the Vite URL, usually `http://127.0.0.1:5173/`.

## Minimal API Example

Observe a window:

```powershell
$body = @{
  hwnd = 123456
  include_screenshot = $false
  allow_vlm = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/observe `
  -ContentType application/json `
  -Body $body
```

Query a returned canvas:

```powershell
$body = @{
  canvas_id = "snap_example"
  query = "primary action"
  limit = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/query `
  -ContentType application/json `
  -Body $body
```

Example JSON payloads are in [examples/api](examples/api).

## Configuration

The public repository does not include local model paths, API keys, screenshots, runtime databases, or real test artifacts.

Start from:

- [config/examples/models.example.yaml](config/examples/models.example.yaml)

Use environment variables for provider keys and keep secrets out of git.

## Privacy

The private development workflow generates screenshots, OCR text, VLM prompts/responses, runtime databases, and real-app reports. Those are intentionally ignored and not included in this public branch.

Ignored by default:

- `artifacts/`
- `data/`
- screenshots and crops
- `.env` files
- runtime databases
- raw VLM payloads

## Tests

Focused smoke tests:

```powershell
python -m pytest tests\unit\test_analyze_goal_progress.py tests\unit\test_analyze_sample_coverage.py tests\unit\test_run_agent_operability_regression.py tests\unit\test_risk_policy.py -q
```

Full real-app tests require a Windows desktop session and local applications, so they are not expected to pass in a generic CI environment.

## License

MIT. See [LICENSE](LICENSE).

