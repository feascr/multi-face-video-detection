# Video Detection Pipeline

Detects faces and ID-like cards in videos using OWLv2 and produces a per‑video decision about whether there are too many uncovered faces.

## Requirements

- Python 3.13.7+
- ffmpeg (required by OpenCV for video IO; preinstalled in the Docker image)

## Quick Start

1) Configure

- Edit `config/config.json` (see Config). At minimum set `data_path`.

2) Install dependencies (choose one)
    - pip:
        - `pip install -r requirements.txt`
    - uv (uses `pyproject.toml`/`uv.lock`):
        - Create venv and sync locked deps: `uv sync`

3) Run

- `python -u main.py`

## Docker

- Build (CPU/GPU): `docker build -t multi-face-detection .`
- CPU run:
  - `docker run --rm --ipc=host --shm-size=2gb -v $(pwd)/config:/app/config -v /path/to/videos:/data:ro -v $(pwd)/output:/app/output multi-face-detection`
- GPU run (NVIDIA toolkit required):
  - `docker run --rm --ipc=host --shm-size=2gb --runtime=nvidia --gpus all -v $(pwd)/config:/app/config -v /path/to/videos:/data:ro -v $(pwd)/output:/app/output multi-face-detection`

## Config

- File: `config/config.json` (overrides defaults in `main.py`).
- Keys:
  - `model_id`: Hugging Face model id (OWLv2 by default)
  - `labels`: either `["label_a", "label_b", ...]` or `[[...]]` for multi-prompt
  - `data_path`: directory with input videos (recursively scanned)
  - `out_solution_path`: TSV path for per-video results (default `output/result.csv`)
  - `detection_results_path`: JSON with per-frame detections (default `output/interim.json`)
  - `log_path`: log file (default `output/log.txt`)
  - `device`: `"cpu"` or `"cuda"`
  - `score_threshold`: score filter after post-processing
  - `nms_threshold`: IOU threshold for per-label NMS
  - `coverage_tolerance`: allowed uncovered fraction when checking if a face is inside a document (0.15 means ≥85% covered counts as inside)
  - `seconds_between_frames`: frame sampling stride in seconds
  - `batch_size`, `num_workers`: DataLoader settings (see Notes)

## Outputs

- `output/interim.json`: per-frame detections per video
- `output/result.csv`: TSV with per-video decision
