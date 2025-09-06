from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.modeling_outputs import BaseModelOutput

from .model_utils import nms_per_label


def init_preprocessor_and_model(model_id: str, device: torch.device):
    """Load OWLv2 processor and model, and move model to `device`."""
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    return processor, model


class ModelTorchInference:
    """Thin inference wrapper around an OWLv2 object detection model.

    Handles text prompt preparation, forward pass, and per-label NMS.
    """

    def __init__(
        self,
        model: Callable,
        preprocessor: Callable,
        postproccessor: Callable,
        label_list: list[list[str]],
        device: torch.device,
        score_threshold: float = 0.1,
        nms_threshold: float = 0.3,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postproccessor
        self.label_list = label_list
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.device = device

    def _forward_model(
        self,
        preprocessed_img_batch: torch.Tensor,
        label_list: list[list[str]],
    ) -> BaseModelOutput:
        with torch.no_grad():
            inputs = self.preprocessor(text=label_list, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = preprocessed_img_batch.to(self.device)
            outputs = self.model(**inputs)
        return outputs

    def _process_model_output(
        self,
        model_output: BaseModelOutput,
        n_imgs: int,
        img_target_sizes: Iterable[tuple[int, int]] | torch.Tensor,
        label_list: list[list[str]],
    ):
        with torch.no_grad():
            results = self.postprocessor(
                outputs=model_output,
                target_sizes=img_target_sizes,
                threshold=self.score_threshold,
                text_labels=label_list,
            )
            # Retrieve predictions for the first image for the corresponding text queries
            all_dets: list[list[dict[str, Any]]] = []

            for i in range(n_imgs):
                res_i = results[i]

                # Bring to CPU NumPy once
                boxes = res_i["boxes"].detach().cpu().numpy()  # (K, 4) float
                scores = res_i["scores"].detach().cpu().numpy()  # (K,)   float
                text_labels = res_i["text_labels"]  # List[str] length K

                # Group per label (fast + simple)
                boxes_by_label = defaultdict(list)  # Dict[str, List[np.ndarray]]
                scores_by_label = defaultdict(list)  # Dict[str, List[float]]

                for b, s, lbl in zip(boxes, scores, text_labels):
                    # score filtering already applied by post_process; keep a guard anyway:
                    if s >= self.score_threshold:
                        boxes_by_label[lbl].append(b)
                        scores_by_label[lbl].append(float(s))

                # Materialize arrays per label (no redundant restacking later)
                preds_per_label = {}
                for lbl in boxes_by_label.keys():
                    bs = np.stack(boxes_by_label[lbl], axis=0) if boxes_by_label[lbl] else np.zeros((0, 4), np.float32)
                    sc = np.asarray(scores_by_label[lbl], dtype=np.float32)
                    preds_per_label[lbl] = {"boxes": bs, "scores": sc}

                # NMS per label -> returns arrays aligned: (M,4), (M,), (M,)
                boxes_nms, scores_nms, labels_nms = nms_per_label(preds_per_label, iou_thresh=self.nms_threshold)
                bboxes_int = np.rint(boxes_nms).astype(np.int32, copy=False)

                # Build list[dict] for this image
                dets_i: list[dict[str, Any]] = [
                    {
                        "bbox": boxes_nms[j].tolist(),  # [x1,y1,x2,y2]
                        "score": float(scores_nms[j]),
                        "label": str(labels_nms[j]),
                        "bbox_int": bboxes_int[j].tolist(),  # [x1,y1,x2,y2]
                    }
                    for j in range(len(scores_nms))
                ]

                all_dets.append(dets_i)
        return all_dets

    def inference_model(
        self,
        preprocessed_img_batch: torch.Tensor,
        img_target_sizes: Iterable[tuple[int, int]] | torch.Tensor,
        label_list: list[list[str]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        label_list = label_list if label_list and len(label_list) else self.label_list

        with torch.no_grad():
            n_imgs = preprocessed_img_batch.shape[0]

        label_list = label_list * n_imgs

        model_output = self._forward_model(preprocessed_img_batch, label_list)
        detections = self._process_model_output(model_output, n_imgs, img_target_sizes, label_list)

        return detections
