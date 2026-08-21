# AI Home Surveillance

A modular AI-powered home surveillance system, built to run on a laptop during development
and deploy to an NVIDIA Jetson Nano in production.

## Status

Project scaffolding only. No camera capture, detection, tracking, alerting, or dashboard
functionality has been implemented yet.

## Project Structure

```
src/
  camera/       # Camera capture and stream handling
  detection/    # Object/person detection models and inference
  tracking/     # Multi-object tracking across frames
  events/       # Event generation, alerting, notifications
  storage/      # Persistence: recordings, clips, event logs
  config/       # Configuration loading and app settings
tests/          # Test suite
scripts/        # Standalone utility / setup scripts
models/         # Model weights (not tracked in git)
data/           # Runtime data output (not tracked in git)
logs/           # Application logs (not tracked in git)
```

## Setup

```bash
python -m venv .venv
# activate the venv for your shell, then:
pip install -r requirements.txt
cp .env.example .env
```

## Design Goals

- Modular: each pipeline stage (camera, detection, tracking, events, storage) is an
  independent, swappable component.
- Portable: the same codebase targets both a development laptop and an NVIDIA Jetson Nano.
- Incremental: functionality is added in stages, starting from a clean structure.
