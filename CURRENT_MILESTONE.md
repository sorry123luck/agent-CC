# Current Milestone

## Goal

Make the DeskCanvas page model usable by external agents without embedding task intent or private app workflows into the service.

## Acceptance Criteria

- `observe` returns a stable `InteractionCanvas` with candidates, regions, OCR/UIA/VLM evidence, and risk metadata.
- `query` can find actionable candidates without relying on raw screenshots.
- page operability reports explain what is ready, what needs review, and what is blocked.
- controlled click/scroll actions can be preflighted, executed, and verified with readback/diff.
- input and send-like actions stay blocked unless a dedicated safety probe confirms the target and state.
- private live artifacts remain outside git; only sanitized reports and fixtures are published.

## Next Work Packages

1. Public sample fixtures: derive small synthetic fixtures from private live results.
2. Agent execution API: harden action categories and verification summaries.
3. Input safety: stabilize composer/input candidates, enabled state, and post-input state changes.
4. VLM semantics: keep ROI use small, parallel, and evidence-bound.
5. Multi-page exploration: record state transitions without defaulting to dangerous actions.

