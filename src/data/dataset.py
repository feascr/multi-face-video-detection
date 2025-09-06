import logging
import math
from pathlib import Path
from typing import Callable, Iterator

import cv2
import torch


class VideoDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that streams frames from videos at fixed intervals.

    - Walks `dataset_path` recursively and tries to open every file as a video.
    - For each valid video, yields frames every `seconds_between_frames` seconds.
    - Optionally applies an image `img_preprocessor` (e.g., HF processor) and
      returns the processed tensor as `processed_frame`.

    Yielded sample dict keys:
      - `video_path`: str, absolute path to the video file
      - `frame_idx`: int, index of the frame (0-based)
      - `target_size`: Tensor[int,int], original (H, W) of the RGB frame
      - `processed_frame`: Tensor[C,H,W]

    """

    def __init__(
        self,
        dataset_path: str,
        seconds_between_frames: float,
        img_preprocessor: Callable,
    ):
        # PyTorch IterableDataset initialization
        super().__init__()
        self.seconds_between_frames = seconds_between_frames

        self.dataset_path = dataset_path
        self.preprocessor = img_preprocessor
        files = self._collect_video_dataset(dataset_path)
        self.video_data = []
        for i in files:
            try:
                metadata = self._validate_video_and_get_metadata(i)
                self.video_data.append((i, metadata))
            except Exception:
                logging.warning(f"Video was not processed: {str(i)}")

    @staticmethod
    def _collect_video_dataset(dataset_path: str) -> list[Path]:
        """Collect candidate video files recursively under `dataset_path`.

        All files are returned; non-video files will be filtered out later when
        attempting to open with OpenCV.
        """
        video_files = []
        for i in Path(dataset_path).rglob("*"):
            if i.is_file():
                video_files.append(i)
        return video_files

    @staticmethod
    def _validate_video_and_get_metadata(
        video_file: Path,
    ) -> dict[str, int | float]:
        """Validate that `video_file` is readable and collect basic metadata.

        Returns a dict with `rotation`, `fps`, and `total_frames`.
        Rotation is best-effort (based on OpenCV metadata) and may be 0.
        """

        def _get_video_metadata(
            cap: cv2.VideoCapture,
        ) -> tuple[int, float, int]:
            rotation = 0
            val = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
            if val == val:  # not NaN
                rotation = int(val) % 360
                rotation = rotation if rotation in (0, 90, 180, 270) else 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return rotation, fps, total_frames

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_file}")
        try:
            rotation, fps, total_frames = _get_video_metadata(cap)
        finally:
            cap.release()

        return {"rotation": rotation, "fps": fps, "total_frames": total_frames}

    @staticmethod
    def _rotate_if_needed(frame, rotation: int):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def _split_for_worker(data, worker_id, num_workers):
        n = len(data)
        per_worker = int(math.ceil(n / num_workers))
        start = worker_id * per_worker
        end = min(start + per_worker, n)
        return data[start:end]

    def __iter__(self) -> Iterator[dict]:
        # Shard videos across DataLoader workers (avoids duplication with num_workers>0)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            videos_to_iterate = self.video_data
        else:
            videos_to_iterate = self._split_for_worker(
                data=self.video_data,
                worker_id=worker_info.id,
                num_workers=worker_info.num_workers,
            )

        for video_path, metadata in videos_to_iterate:
            # rotation = metadata["rotation"]
            fps = metadata["fps"]
            total_frames = metadata["total_frames"]

            step = max(1, int(round(self.seconds_between_frames * fps)))

            current_idx = 0

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")
            try:
                while current_idx < total_frames:
                    # Fetch frames at those indices (random access via CAP_PROP_POS_FRAMES)
                    data = {}

                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_idx))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        # If a read fails, stop collecting this batch at that point
                        break
                    # if rotation:
                    #     frame = self._rotate_if_needed(frame, rotation)
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    data["target_size"] = torch.tensor(processed_frame.shape[:2])
                    processed_frame = self.preprocessor(images=processed_frame, return_tensors="pt")
                    processed_frame = processed_frame["pixel_values"].squeeze(0)
                    data["processed_frame"] = processed_frame

                    data["video_path"] = str(video_path)
                    data["frame_idx"] = current_idx

                    current_idx += step
                    yield data
            finally:
                cap.release()
