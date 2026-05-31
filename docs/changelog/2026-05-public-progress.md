# 2026-05 Public Progress

This is a sanitized changelog summary for the public repository. Detailed private live artifacts are not included.

## Recognition and VLM

- Added ROI-level VLM supplement workers with late-result handling.
- Added prompt profiles and ROI budgets for dense controls, chat composers, and unknown icon regions.
- Added gates so VLM evidence supplements candidates without overwriting strong local evidence.
- Added diagnostics for VLM result binding, timeout behavior, and wrong-region projection.

## Page Model and Operability

- Added page operability reports for agent-facing readiness.
- Added read-region crop/OCR support.
- Added sample coverage and recognition closure reports.
- Added transition readiness reporting for future multi-page exploration.

## Execution Safety

- Added `/api/v1/act` preflight and policy semantics.
- Added controlled click/scroll execution with verification records.
- Added readback status and action transition persistence.
- Kept typing and send-like actions behind stricter gates.

## Privacy Cleanup

- Public branch excludes raw artifacts, screenshots, runtime databases, and private live sample evidence.
- Public docs summarize progress without including private target names or message contents.

