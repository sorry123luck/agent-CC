# Privacy and Artifact Policy

The development workflow produces screenshots, OCR text, VLM prompts, VLM responses, window titles, coordinates, runtime databases, and timing reports. These are useful for debugging but may contain private data.

## Not Published

- `artifacts/`
- `data/`
- runtime logs
- local model paths
- API keys and `.env` files
- screenshots and image crops
- raw VLM requests/responses
- real chat/window evidence

## Published

- source code
- unit/e2e test harnesses
- sanitized progress summaries
- public architecture docs
- public changelog summaries
- example configuration with placeholders

## Contributor Rule

Before committing, run a leak check for local paths, secrets, screenshots, runtime databases, and raw artifacts.

