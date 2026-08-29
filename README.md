# AI Home Surveillance

A modular AI-powered home surveillance system. It is developed on a laptop today
and is intended to run on an NVIDIA Jetson Nano later.

## Current capabilities

The full pipeline exists end to end: a camera feed is detected, tracked, turned
into zone enter/exit events, recorded to clips and JSON metadata, alerted on,
and made browsable through a web dashboard. Every stage isolates its own
failures, so a bad frame, a full disk, a dropped camera, or a broken alert
handler cannot bring the surveillance loop down.

| Capability | State |
| --- | --- |
| Camera capture (webcam, RTSP, video file) | Working |
| Camera failure handling and reconnect | Working |
| Person detection with YOLO | Working |
| Multi-object tracking | Working |
| Zone enter/exit events | Working |
| Recording (video clips + snapshots) | Working |
| Event metadata storage + retention | Working |
| Alerts (console handler, pluggable) | Working |
| Pipeline fault isolation (per stage) | Working |
| Structured application logging | Working |
| Read-only HTTP API | Working |
| Web dashboard | Working |
| Configuration via environment variables | Working |
| Unit tests (no camera, GPU or weights needed) | Working — 310 tests |
| **Jetson Nano deployment, TensorRT** | **Not started** |

## Architecture

Each stage is an independent module; nothing imports "sideways" except through
`SurveillancePipeline`, which is the only place they are wired together:

```
src/pipeline/pipeline.py     <- composition root: owns the run loop and fault isolation
        |
        +-- Camera().read()             src/camera/     owns cv2.VideoCapture, reconnect
        +-- Detector().detect()         src/detection/  owns Ultralytics YOLO
        +-- Tracker().update()          src/tracking/   assigns stable track IDs
        +-- EventEngine().update()      src/events/     zone enter/exit, knows nothing else
        +-- VideoRecorder                src/storage/    clips + snapshots, owns cv2.VideoWriter
        +-- MetadataStore                src/storage/    one JSON file per event
        +-- AlertDispatcher              src/alerts/     fans an event out to handlers

src/api/app.py                <- read-only FastAPI app over MetadataStore + StorageManager
        +-- src/api/static/index.html   <- vanilla HTML/CSS/JS dashboard, no build step
```

- `Camera` is the only place `cv2.VideoCapture` is used, and the only place
  that knows how to reconnect after a read failure.
- `Detector` is the only place Ultralytics is used. It returns `Detection`
  objects built from plain Python types, so nothing downstream imports YOLO.
- `EventEngine` only ever sees `TrackedObject` data and emits `Event` objects;
  it has no idea recording, storage or alerting exist.
- `SurveillancePipeline` is the one place that turns events into recordings,
  metadata and alerts. Each stage call is isolated: a detector, tracker, event
  engine, recorder, storage, or alert failure is logged and the loop continues.
- The dashboard and API are read-only: they only read what the pipeline
  already wrote through `MetadataStore`/`StorageManager`, so they can run on a
  machine with no camera at all.
- Settings are read once at startup and passed in as arguments. Modules do not
  read environment variables themselves.

## Repository structure

```
src/
  main.py         Application entry point (logging + config summary)
  camera/         Camera capture, read failures, reconnect
  detection/      YOLO inference and the Detection result type
  tracking/       Multi-object tracking (stable track IDs across frames)
  events/         Zone enter/exit event generation
  storage/        VideoRecorder, MetadataStore, StorageManager (retention)
  alerts/         Alert type + pluggable AlertDispatcher
  pipeline/       Wires camera/detector/tracker/events/storage/alerts together
  api/            Read-only FastAPI app, plus static/ (the dashboard)
  config/         Settings loading and logging setup
tests/            Unit tests (standard-library unittest)
scripts/          Manual demo scripts for laptop development
models/           Model weights (gitignored)
data/             Runtime output: recordings, snapshots, metadata (gitignored)
logs/             Log files (gitignored)
```

## Laptop setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/katakamX/ai-home-surveillance.git
cd ai-home-surveillance

python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate       # Windows PowerShell / cmd
source .venv/bin/activate    # macOS / Linux
```

Install the dependencies. This pulls in PyTorch, so expect a large download:

```bash
pip install -r requirements.txt
```

Configuration is optional. To change any default, copy the example file and edit it:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAMERA_SOURCE` | `0` | Webcam index, RTSP URL, or video file path |
| `MODEL_DIR` | `models` | Where weights are kept |
| `MODEL_PATH` | `models/yolov8n.pt` | Full path to the weights file |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence, 0.0 to 1.0 |
| `PERSON_ONLY` | `true` | Report only people, ignoring other classes |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `DATA_DIR` | `data` | Where recordings, snapshots and event metadata are written |

## Running

All commands are run from the project root.

Entry point — prints the effective configuration and exits (the pipeline
itself is driven from your own script or test; `src/main.py` is a
configuration/logging smoke test, not a long-running process yet):

```bash
python -m src.main
```

Camera demo — shows the live feed. Press `q` in the window to quit:

```bash
python scripts/demo_camera.py
```

Detection demo — draws boxes and confidence scores around people:

```bash
python scripts/demo_detection.py
```

Tracking demo — draws stable track IDs across frames:

```bash
python scripts/demo_tracking.py
```

Events demo — prints zone enter/exit events as they happen:

```bash
python scripts/demo_events.py
```

Dashboard and API — serves the read-only web dashboard at `/` and JSON at
`/events`, over whatever `DATA_DIR` already has recorded:

```bash
uvicorn src.api.app:app --reload
```

The first detection run downloads `yolov8n.pt` (about 6 MB) into `models/`.
On a laptop CPU expect only a few frames per second; that is normal.

## Running the tests

```bash
python -m unittest discover -v
```

310 tests, no camera, no GPU, no model weights and no internet access needed.
OpenCV and Ultralytics are replaced with mocks, so the tests also pass before
`pip install -r requirements.txt` has been run.

## Current limitations

- No long-running service wiring yet: `src/main.py` reports configuration but
  does not start `SurveillancePipeline.run()`. Assembling `Camera` +
  `Detector` + `Tracker` + `EventEngine` + `SurveillancePipeline` into a
  single always-on process (systemd/Task Scheduler style) is still open.
- CPU inference is slow. No GPU or TensorRT acceleration is configured.
- The dashboard has no authentication and no live video streaming; it shows
  stored snapshots, clips and event history only.
- No database — metadata is one JSON file per event, which is fine at home
  camera scale but would not suit a multi-camera fleet.

## Roadmap

1. ~~Project structure~~
2. ~~Camera input~~
3. ~~Person detection~~
4. ~~Tracking — follow the same person across frames~~
5. ~~Events — decide what is worth reporting~~
6. ~~Storage — save clips and an event log~~
7. ~~Alerts — notify on meaningful events~~
8. ~~Dashboard — view the feed and event history~~
9. ~~Camera resilience — survive read failures and reconnect~~
10. ~~Pipeline fault isolation, observability, long-running stability~~
11. **Jetson Nano deployment — aarch64 dependencies, TensorRT, autostart**

## Jetson Nano status

Not implemented. Nothing in this repository is Jetson-specific yet, and
`requirements.txt` should not be installed as-is on the device: JetPack ships its
own OpenCV build, and PyTorch must come from NVIDIA's aarch64 wheels. Treat
deployment as a future stage with its own setup instructions.
