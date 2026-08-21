# AI Home Surveillance

A modular AI-powered home surveillance system. It is developed on a laptop today
and is intended to run on an NVIDIA Jetson Nano later.

## Current capabilities

The project can currently capture frames from a camera and detect people in them.
That is all — there is no application pipeline yet, only the building blocks and
two demo scripts that wire them together by hand.

| Capability | State |
| --- | --- |
| Camera capture (webcam, RTSP, video file) | Working |
| Person detection with YOLO | Working |
| Configuration via environment variables | Working |
| Console logging | Working |
| Unit tests (no camera, GPU or weights needed) | Working |
| Tracking, events, alerts, storage | Not started |
| Web dashboard, API | Not started |
| **Jetson Nano deployment, TensorRT** | **Not started** |

## Architecture

Two independent building blocks, joined only at the top level. Neither imports
the other, so either can be replaced on its own:

```
scripts/demo_detection.py          <- composition root: reads settings, wires things together
        |
        +-- Camera(source).read()  -> (success, frame)     src/camera/    owns cv2.VideoCapture
        +-- Detector().detect(frame) -> [Detection]        src/detection/ owns Ultralytics YOLO
```

- `Camera` is the only place `cv2.VideoCapture` is used.
- `Detector` is the only place Ultralytics is used. It returns `Detection`
  objects built from plain Python types (`str`, `float`, `tuple[int, ...]`), so
  the future tracking module will never import YOLO or PyTorch.
- Settings are read once at startup and passed in as arguments. The modules do
  not read environment variables themselves.

## Repository structure

```
src/
  main.py         Application entry point (logging + config summary for now)
  camera/         Camera capture and stream handling
  detection/      YOLO inference and the Detection result type
  config/         Settings loading and logging setup
  tracking/       Multi-object tracking            (empty, future stage)
  events/         Event generation and alerting    (empty, future stage)
  storage/        Recordings and event persistence (empty, future stage)
tests/            Unit tests (standard-library unittest)
scripts/          Manual demo scripts for laptop development
models/           Model weights (gitignored)
data/             Runtime output (gitignored)
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
| `DATA_DIR` | `data` | Where recordings will go once storage exists |

## Running

All commands are run from the project root.

Entry point — prints the effective configuration and exits:

```bash
python -m src.main
```

Camera demo — shows the live feed. Press `q` in the window to quit:

```bash
python scripts/demo_camera.py
```

Detection demo — draws boxes and confidence scores around people. Press `q` to quit:

```bash
python scripts/demo_detection.py
```

The first detection run downloads `yolov8n.pt` (about 6 MB) into `models/`.
On a laptop CPU expect only a few frames per second; that is normal.

## Running the tests

```bash
python -m unittest discover -v
```

The suite needs no camera, no GPU, no model weights and no internet access.
OpenCV and Ultralytics are replaced with mocks, so the tests also pass before
`pip install -r requirements.txt` has been run.

## Current limitations

- No pipeline: `src/main.py` does not capture or detect anything. Only the demo
  scripts do, and they are development tools, not production code.
- Detection is per-frame only. Nothing links a person in one frame to the same
  person in the next.
- Nothing is recorded, stored or alerted on.
- No reconnection logic. If an RTSP stream drops, the demo stops.
- CPU inference is slow. No GPU or TensorRT acceleration is configured.
- `tracking/`, `events/` and `storage/` are empty placeholders.

## Roadmap

1. ~~Project structure~~
2. ~~Camera input~~
3. ~~Person detection~~
4. Tracking — follow the same person across frames
5. Events — decide what is worth reporting
6. Storage — save clips and an event log
7. Alerts — notify on meaningful events
8. Dashboard — view the feed and event history
9. **Jetson Nano deployment — aarch64 dependencies, TensorRT, autostart**

## Jetson Nano status

Not implemented. Nothing in this repository is Jetson-specific yet, and
`requirements.txt` should not be installed as-is on the device: JetPack ships its
own OpenCV build, and PyTorch must come from NVIDIA's aarch64 wheels. Treat
deployment as a future stage with its own setup instructions.
