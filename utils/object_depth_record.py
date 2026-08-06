"""Per-object depth records from YOLO detections + GT depth maps."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from utils.object_detection_cache import ObjectDetection, _stem_and_parent, load_detections


CSV_FIELDS = [
    "class_id",
    "class_name",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "bbox_cy_norm",
    "depth_gt_m",
    "disparity_gt",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "dataset",
    "depth_unit",
    "schema_version",
    "mask_fill_ratio",
    "mask_axis_ratio",
    "mask_angle_sin2",
    "mask_angle_cos2",
    "mask_orientation_confidence",
    "mask_valid",
]

OBJECT_RECORD_SCHEMA_VERSION = 3
DEPTH_UNIT_DIVISORS = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


@dataclass
class ObjectDepthRecord:
    class_id: int
    class_name: str
    bbox_w: float
    bbox_h: float
    bbox_area: float
    bbox_cy_norm: float
    depth_gt_m: float
    disparity_gt: float
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    dataset: str = "unknown"
    depth_unit: str = "m"
    schema_version: int = OBJECT_RECORD_SCHEMA_VERSION
    mask_fill_ratio: float = 1.0
    mask_axis_ratio: float = 1.0
    mask_angle_sin2: float = 0.0
    mask_angle_cos2: float = 1.0
    mask_orientation_confidence: float = 0.0
    mask_valid: float = 0.0

    def to_row(self) -> dict:
        return asdict(self)


def objects_csv_path(rgb_path: str | Path, detail_root: str | Path) -> Path:
    detail_root = Path(detail_root)
    parent, stem = _stem_and_parent(rgb_path)
    return detail_root / parent / f"{stem}_objects.csv"


def load_gt_depth_meters(depth_path: str | Path, *, depth_unit: str) -> np.ndarray:
    """Decode a depth image using an explicit source unit.

    Value-based unit inference is intentionally forbidden because uint16 alone
    does not identify whether a dataset stores millimetres or centimetres.
    """
    if depth_unit not in DEPTH_UNIT_DIVISORS:
        raise ValueError(
            f"Unsupported depth_unit={depth_unit!r}; expected one of "
            f"{sorted(DEPTH_UNIT_DIVISORS)}"
        )
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    depth /= DEPTH_UNIT_DIVISORS[depth_unit]
    return np.clip(depth, 1e-4, None)


def guess_depth_path(rgb_path: str | Path) -> Path:
    p = Path(rgb_path)
    if p.parent.name == "color":
        candidate = p.parent.parent / "depth" / f"{p.stem}.png"
        if candidate.is_file():
            return candidate
    if p.name.startswith("rgb_cam_"):
        candidate = p.parent / p.name.replace("rgb_cam_", "depth_plane_cam_")
        if candidate.is_file():
            return candidate
    if p.name.startswith("rgb_"):
        candidate = p.parent / p.name.replace("rgb_", "depth_", 1)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not infer depth path for {rgb_path}")


def depth_median_in_bbox(
    depth_m: np.ndarray,
    bbox: Sequence[int],
    *,
    min_pixels: int = 16,
) -> Optional[float]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth_m.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_m[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 1e-4)
    if int(valid.sum()) < min_pixels:
        return None
    return float(np.median(patch[valid]))


def records_from_detections(
    detections: Sequence[ObjectDetection],
    depth_m: np.ndarray,
    *,
    score_thr: float = 0.5,
    min_pixels: int = 16,
    dataset: str,
    depth_unit: str,
) -> List[ObjectDepthRecord]:
    h, w = depth_m.shape[:2]
    img_w = max(int(w), 1)
    img_h = max(int(h), 1)
    records: List[ObjectDepthRecord] = []

    for det in detections:
        if det.score < score_thr:
            continue
        x1, y1, x2, y2 = det.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        depth_gt = depth_median_in_bbox(depth_m, det.bbox, min_pixels=min_pixels)
        if depth_gt is None:
            continue
        bw = float(x2 - x1) / float(img_w)
        bh = float(y2 - y1) / float(img_h)
        cy = float(y1 + y2) * 0.5 / float(img_h)
        records.append(
            ObjectDepthRecord(
                class_id=int(det.class_id),
                class_name=str(det.class_name),
                bbox_w=bw,
                bbox_h=bh,
                bbox_area=bw * bh,
                bbox_cy_norm=cy,
                depth_gt_m=depth_gt,
                disparity_gt=1.0 / depth_gt,
                score=float(det.score),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                dataset=dataset,
                depth_unit=depth_unit,
                schema_version=OBJECT_RECORD_SCHEMA_VERSION,
                mask_fill_ratio=float(det.mask_fill_ratio),
                mask_axis_ratio=float(det.mask_axis_ratio),
                mask_angle_sin2=float(det.mask_angle_sin2),
                mask_angle_cos2=float(det.mask_angle_cos2),
                mask_orientation_confidence=float(det.mask_orientation_confidence),
                mask_valid=float(det.mask_valid),
            )
        )
    return records


def build_records_for_image(
    rgb_path: str | Path,
    detail_root: str | Path,
    depth_path: str | Path | None = None,
    *,
    score_thr: float = 0.5,
    min_pixels: int = 16,
    dataset: str,
    depth_unit: str,
) -> List[ObjectDepthRecord]:
    depth_path = Path(depth_path) if depth_path is not None else guess_depth_path(rgb_path)
    detections = load_detections(rgb_path, detail_root)
    depth_m = load_gt_depth_meters(depth_path, depth_unit=depth_unit)
    return records_from_detections(
        detections,
        depth_m,
        score_thr=score_thr,
        min_pixels=min_pixels,
        dataset=dataset,
        depth_unit=depth_unit,
    )


def save_object_records(
    records: Sequence[ObjectDepthRecord],
    rgb_path: str | Path,
    detail_root: str | Path,
) -> Path:
    out_path = objects_csv_path(rgb_path, detail_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_row())
    return out_path


def load_object_records(csv_path: str | Path) -> List[ObjectDepthRecord]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return []
    rows: List[ObjectDepthRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_metadata = {"dataset", "depth_unit", "schema_version"}
        missing = required_metadata.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Legacy object record schema in {csv_path}; missing {sorted(missing)}. "
                "Regenerate with utils/build_object_depth_records.py."
            )
        for row in reader:
            schema_version = int(row["schema_version"])
            if schema_version not in (2, OBJECT_RECORD_SCHEMA_VERSION):
                raise ValueError(
                    f"Unsupported object record schema {schema_version} in {csv_path}; "
                    f"expected 2 or {OBJECT_RECORD_SCHEMA_VERSION}."
                )
            rows.append(
                ObjectDepthRecord(
                    class_id=int(row["class_id"]),
                    class_name=str(row["class_name"]),
                    bbox_w=float(row["bbox_w"]),
                    bbox_h=float(row["bbox_h"]),
                    bbox_area=float(row.get("bbox_area") or float(row["bbox_w"]) * float(row["bbox_h"])),
                    bbox_cy_norm=float(row.get("bbox_cy_norm") or 0.0),
                    depth_gt_m=float(row["depth_gt_m"]),
                    disparity_gt=float(row["disparity_gt"]),
                    score=float(row["score"]),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                    x2=int(row["x2"]),
                    y2=int(row["y2"]),
                    dataset=str(row["dataset"]),
                    depth_unit=str(row["depth_unit"]),
                    schema_version=schema_version,
                    mask_fill_ratio=float(row.get("mask_fill_ratio") or 1.0),
                    mask_axis_ratio=float(row.get("mask_axis_ratio") or 1.0),
                    mask_angle_sin2=float(row.get("mask_angle_sin2") or 0.0),
                    mask_angle_cos2=float(row.get("mask_angle_cos2") or 1.0),
                    mask_orientation_confidence=float(
                        row.get("mask_orientation_confidence") or 0.0
                    ),
                    mask_valid=float(row.get("mask_valid") or 0.0),
                )
            )
    return rows


def load_object_records_for_rgb(rgb_path: str | Path, detail_root: str | Path) -> List[ObjectDepthRecord]:
    return load_object_records(objects_csv_path(rgb_path, detail_root))


def iter_object_record_files(detail_root: str | Path) -> Iterable[Path]:
    detail_root = Path(detail_root)
    yield from sorted(detail_root.rglob("*_objects.csv"))
