# Status

This public status file is a sanitized summary. Private screenshots, real chat samples, window evidence, and raw runtime artifacts are intentionally excluded from this branch.

## Current Phase

DeskCanvas is in the agent-operability phase:

- page observation and candidate generation are usable for inspection;
- VLM supplements are treated as semantic evidence, not as the source of coordinates;
- page operability reports aggregate canvas, regions, candidates, read regions, evidence, and risk;
- guarded action preflight supports policy classes for read-only, review, controlled, and blocked actions;
- controlled click/scroll paths include readback/diff verification;
- text input and send-like actions remain blocked unless a target-specific safety gate is satisfied.

## Completed Public Milestones

- Core `InteractionCanvas` model and query/diff/memory APIs.
- Windows observation through UIA, OCR, geometric partitioning, and optional VLM ROI supplement.
- React console for inspecting windows, canvases, candidates, evidence, and VLM settings.
- ROI VLM worker and prompt profiles with late result retention.
- Local evidence gates to avoid sending text-labeled controls to VLM unnecessarily.
- Dynamic content gates so message streams and document content do not pollute persistent control semantics.
- Page operability and regression reports for agent-facing readiness.
- `/api/v1/act` preflight and guarded action execution policy.
- Read-region crop/OCR and scroll/readback test harnesses.

## Open Work

- Improve general input-zone safety before allowing any broad `type_text` or send action.
- Continue reducing unknown/icon-only controls through local rules plus small ROI VLM batches.
- Promote private live sample matrices into sanitized public fixtures.
- Expand multi-page exploration after single-page operability stabilizes.
- Add CI-friendly smoke tests that do not require a live Windows desktop.

