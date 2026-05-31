# Contributing

OpenClaw is still a prototype. Keep changes small, tested, and privacy-aware.

## Before Opening a PR

1. Run focused unit tests for the area you touched.
2. Do not commit `artifacts/`, `data/`, screenshots, local configs, model weights, or API keys.
3. Document behavior changes in `docs/changelog/`.
4. Keep app-specific live samples out of public fixtures unless they are synthetic or fully sanitized.

## Development Notes

- Perception should prefer local evidence before VLM.
- VLM should add semantic evidence, not replace coordinates from local candidates.
- Execution code must preserve policy classes and verification records.
- Text input and send-like actions require stricter review than click/scroll.

