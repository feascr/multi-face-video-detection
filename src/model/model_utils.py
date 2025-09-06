import numpy as np


def nms_per_label(preds_per_label, iou_thresh=0.5):
    kept_boxes = []
    kept_scores = []
    kept_labels = []

    for label, pred_dict in preds_per_label.items():
        b, s = pred_dict["boxes"], pred_dict["scores"]
        if b.size == 0:
            continue
        keep_idx = nms(b, s, iou_thresh)
        kept_boxes.extend(b[keep_idx])
        kept_scores.extend(s[keep_idx])
        kept_labels.extend([label] * len(keep_idx))

    if kept_boxes:
        kept_boxes = np.stack(kept_boxes, axis=0)
        kept_scores = np.array(kept_scores, dtype=np.float32)
    else:
        kept_boxes = np.zeros((0, 4), dtype=np.float32)
        kept_scores = np.zeros((0,), dtype=np.float32)

    return kept_boxes, kept_scores, kept_labels


def nms(boxes, scores, iou_thresh):
    if boxes.size == 0:
        return []

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-9)

        remain = np.where(iou <= iou_thresh)[0]
        order = order[remain + 1]

    return keep
