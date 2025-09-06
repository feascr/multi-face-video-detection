def _area(b) -> float:
    """Axis-aligned box area for [x1, y1, x2, y2]."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_dims(b1, b2):
    """Intersection width and height for two [x1, y1, x2, y2] boxes."""
    w = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    h = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    return w, h


def coverage(b_inner, b_outer) -> float:
    """Fraction of `b_inner` covered by `b_outer` (0..1)."""
    w, h = _intersection_dims(b_inner, b_outer)
    inter = w * h
    denom = _area(b_inner)
    return inter / denom if denom > 0 else 0.0


def is_inside_coverage(b1, b2, tolerance: float = 0.15) -> bool:
    """Return True if at least (1 - tolerance) of `b1` is covered by `b2`."""
    cov = coverage(b1, b2)
    return cov >= (1 - tolerance)


def is_at_least_one_inside(bbox_list1, bbox_list2, *, tolerance: float = 0.15):
    """Count boxes from `bbox_list1` not covered by any box in `bbox_list2`.

    Returns:
      - num_faces_left: int, count of boxes in list1 with zero covering matches
      - inside_res: list, kept for backward-compatibility (currently unused)
      - inside_idxs: list[tuple[int,int]], pairs of indices (i, j) that matched
    """
    inside_res = []
    inside_idxs = []
    num_faces_left = 0

    for i, bbox1 in enumerate(bbox_list1):
        covered = 0
        for j, bbox2 in enumerate(bbox_list2):
            inside = is_inside_coverage(bbox1, bbox2, tolerance=tolerance)
            if inside:
                inside_idxs.append((i, j))
                covered += 1
        if covered == 0:
            num_faces_left += 1

    return num_faces_left, inside_res, inside_idxs


def check_inside_for_labels(
    data,
    label1: str = "a photo of a face",
    label2: str = "id or card or rectangular item",
    *,
    tolerance: float = 0.15,
):
    """Extract bboxes by label and check coverage of `label1` within `label2`.

    - Filters by label only (score filtering handled upstream by the model).
    - Uses `tolerance` as the allowed uncovered fraction of `label1` boxes.

    Returns the tuple from `is_at_least_one_inside`.
    """
    bbox_list1 = [item["bbox_int"] for item in data["bboxes"] if item["label"] == label1]
    bbox_list2 = [item["bbox_int"] for item in data["bboxes"] if item["label"] == label2]
    return is_at_least_one_inside(bbox_list1, bbox_list2, tolerance=tolerance)
