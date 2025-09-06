"""End-to-end video scanning pipeline using OWLv2 object detection.

Reads configuration, iterates frames from videos, runs detection, writes
intermediate detections and a final per-video CSV with a simple decision.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import check_inside_for_labels, configure_logger, seed_everything
from src.data import VideoDataset
from src.model import ModelTorchInference, init_preprocessor_and_model

DEFAULT_CONFIG = {
    "model_id": "google/owlv2-base-patch16-ensemble",
    "labels": ["a photo of a face", "id or card or rectangular item"],
    "data_path": "",
    "out_solution_path": "output/result.csv",
    "detection_results_path": "output/interim.json",
    "log_path": "output/log.txt",
    "device": "cuda",
    "score_threshold": 0.15,
    "nms_threshold": 0.3,
    "coverage_tolerance": 0.15,
    "seed": 9836,
    "seconds_between_frames": 1.0,
    "batch_size": 4,
    "num_workers": 4,
}


def load_config(path: str = "config/config.json") -> dict:
    """Load config JSON and merge with `DEFAULT_CONFIG`.

    Values in the file override defaults. Missing file falls back to defaults.
    """
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    merged = {**DEFAULT_CONFIG, **cfg}
    return merged


def main() -> None:
    """Run the pipeline according to configuration."""
    parser = argparse.ArgumentParser(description="OWLv2 video scanning pipeline")
    parser.add_argument(
        "-c",
        "--config",
        default="config/config.json",
        help="Path to configuration JSON file (default: config/config.json)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    labels_value = cfg.get("labels", [])
    labels = labels_value if (labels_value and isinstance(labels_value[0], list)) else [labels_value]
    data_path = cfg["data_path"]
    out_solution_path = cfg["out_solution_path"]
    detection_results_path = cfg["detection_results_path"]
    log_path = cfg["log_path"]
    device_arg = cfg["device"]
    score_threshold = float(cfg["score_threshold"])
    nms_threshold = float(cfg["nms_threshold"])
    seed = int(cfg["seed"])
    coverage_tolerance = float(cfg["coverage_tolerance"])

    # Ensure output directories exist for all configured output artifacts
    for p in (out_solution_path, detection_results_path, log_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    configure_logger(log_path)
    logging.info("Using config file: %s", args.config)

    seed_everything(seed)
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        logging.info("CUDA not available; falling back to CPU")
        device_arg = "cpu"
    device = torch.device(device_arg)
    logging.info("Using device: %s", device)
    preprocessor, model = init_preprocessor_and_model(model_id, device)
    logging.info("Initialized model: %s", model_id)

    logging.info(
        "Loading dataset from %s (every %.2fs)",
        data_path,
        float(cfg["seconds_between_frames"]),
    )
    dataset = VideoDataset(data_path, float(cfg["seconds_between_frames"]), preprocessor)

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )

    model_inference = ModelTorchInference(
        model,
        preprocessor,
        preprocessor.post_process_grounded_object_detection,
        labels,
        device,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )

    results = []
    logging.info("Starting inference")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            video_paths = batch["video_path"]
            frame_idxs = batch["frame_idx"]
            target_size = batch["target_size"]
            frame_batch = batch["processed_frame"].to(device)

            preds = model_inference.inference_model(
                frame_batch,
                target_size,
            )

            # Collect per-frame detections
            for video_p, frame_i, pred_i in zip(video_paths, frame_idxs, preds):
                results.append(
                    {
                        "video": video_p,
                        "frame_idx": frame_i.item(),
                        "bboxes": pred_i,
                    }
                )

    logging.info("Writing detections to: %s", detection_results_path)
    with open(detection_results_path, "w") as f:
        json.dump(results, f)

    per_video_data = defaultdict(list)
    for i in results:
        per_video_data[Path(i["video"]).stem].append(i)

    solution = []
    for k, v in per_video_data.items():
        video_res = []
        for frame in v:
            num_faces_left, _, _ = check_inside_for_labels(
                frame,
                tolerance=coverage_tolerance,
            )
            video_res.append(num_faces_left)
        is_too_many_faces = int(np.any(np.array(video_res) > 1))
        solution.append({"video": k, "prediction": is_too_many_faces})
    df = pd.DataFrame(solution)

    # Enrich with dataset labels from labels.txt if present in dataset root
    labels_txt = Path(data_path) / "labels.txt"
    if labels_txt.exists():
        labels_df = pd.read_csv(labels_txt, sep="\t")
        # Merge on 'video' to append ground-truth label
        df = df.merge(labels_df[["video", "label"]], on="video", how="left")

    logging.info(
        "Saving per-video predictions (%d rows) to: %s",
        len(df),
        out_solution_path,
    )
    df.to_csv(out_solution_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
