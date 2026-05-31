# Codex for OSS Application Notes

Repository: `https://github.com/sorry123luck/agent-CC`

## Short Description

OpenClaw is an open-source Windows desktop perception and guarded action layer for AI agents. It converts real application windows into structured page models with candidates, evidence, coordinates, risk labels, memory, and verification hooks.

## Why It Is Useful

Most desktop agents either rely on raw screenshots or app-specific automations. OpenClaw aims to provide a reusable middle layer: fast local perception, optional VLM semantic completion, persistent page memory, and conservative execution gates that an external agent can consume.

## Current Progress

- Fast observe and page model generation.
- Multi-source perception fusion.
- ROI VLM semantic supplement.
- Visual console for inspecting candidates and evidence.
- Page operability and regression reporting.
- Controlled action preflight and verification.
- Privacy-aware artifact policy for live desktop samples.

## Why Codex Helps

The project needs sustained engineering across perception, safety policy, Windows automation, frontend inspection tools, regression tooling, and documentation. Codex can help keep the architecture coherent while expanding coverage across real desktop applications.

