# Codex for OSS Application Notes

Repository: `https://github.com/sorry123luck/agent-CC`

## One-Line Project Description

OpenClaw is an agent-facing Windows desktop perception and guarded-action API that turns real app windows into structured page models an AI agent can query and act on safely.

## What It Is

OpenClaw is a local service and inspection console for desktop agents. It is not an app-specific automation script and not a chatbot. It provides the agent with:

- window observation;
- structured candidates with coordinates and evidence;
- page regions and dynamic-content boundaries;
- query/diff/memory APIs;
- guarded action preflight and verification;
- privacy-aware testing and artifact policy.

## Why It Matters

Most desktop agents choose between two weak options:

1. operate directly on screenshots and hope the model understands every coordinate;
2. build app-specific scripts that break on unknown software.

OpenClaw aims for a reusable middle layer. It gives the agent a stable UI map while keeping final task reasoning outside the service.

## Current Public Capabilities

- Windows window enumeration and screenshot capture.
- UIA/OCR/geometric fusion into `InteractionCanvas`.
- Optional ROI VLM semantic supplement for ambiguous controls.
- React console for reviewing candidates, evidence, regions, VLM output, and memory.
- Page operability reports for agent readiness.
- `/api/v1/act` policy classes: `read-only`, `review`, `controlled`, `blocked`.
- Controlled click/scroll with observe/diff/readback verification.
- Conservative input/send safety gates.
- Unit, e2e, and real-app test harnesses.

## Showcase Summary

OpenClaw converts desktop UI from raw pixels into agent-readable state:

- raw screenshot -> canvas with regions;
- unknown icon -> candidate with local evidence plus optional VLM semantic label;
- click request -> preflight policy decision;
- action result -> verified transition/readback record.

Private live screenshots and chat samples are intentionally excluded from the public branch. The public repository includes the code, test harnesses, sanitized progress summaries, and example API payloads.

## Why Codex Helps

This project needs sustained engineering across Windows automation, perception fusion, VLM prompt/evidence design, safety policy, regression tooling, and frontend inspection. Codex is a strong fit because the project has a real codebase, a clear agent-facing API, and many incremental engineering tasks that benefit from long-running code assistance.

