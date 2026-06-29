"""Image-to-image and video comparison logic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import ObjectDetector
from .geometry import relative_angle, rotation_direction
from .visualization import draw_label


@dataclass
class ComparisonResult:
    reference_angle: float
    current_angle: float
    relative_rotation: float
    direction: str
    confidence: float
    reference_center: tuple[int, int]
    current_center: tuple[int, int]


def compare_images(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    detector: ObjectDetector,
) -> ComparisonResult | None:
    ref_dets, _ = detector.detect(ref_image)
    cur_dets, _ = detector.detect(cur_image)
    if not ref_dets or not cur_dets:
        return None

    rd, cd = ref_dets[0], cur_dets[0]
    rel = relative_angle(cd.axis_angle, rd.axis_angle)

    return ComparisonResult(
        reference_angle=rd.axis_angle,
        current_angle=cd.axis_angle,
        relative_rotation=rel,
        direction=rotation_direction(rel),
        confidence=min(rd.confidence, cd.confidence),
        reference_center=rd.center,
        current_center=cd.center,
    )


def annotate_comparison(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    result: ComparisonResult,
    decimals: int = 2,
) -> np.ndarray:
    gap = 20
    footer = 50
    h1, w1 = ref_image.shape[:2]
    h2, w2 = cur_image.shape[:2]
    panel_h = max(h1, h2)

    canvas = np.zeros((panel_h + footer, w1 + w2 + gap, 3), dtype=np.uint8)
    canvas[:h1, :w1] = ref_image
    canvas[:h2, w1 + gap:] = cur_image

    footer_y = panel_h + 30
    draw_label(canvas, f"REFERENCE: {result.reference_angle:.{decimals}f}deg", (10, footer_y))
    draw_label(canvas, f"CURRENT: {result.current_angle:.{decimals}f}deg", (w1 + gap + 10, footer_y))
    draw_label(
        canvas,
        f"RELATIVE: {result.relative_rotation:+.{decimals}f}deg "
        f"({result.direction}) Conf:{result.confidence * 100:.0f}%",
        (10, 24),
        color=(0, 255, 255),
    )
    return canvas


def save_comparison_report(result: ComparisonResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "reference_angle": result.reference_angle,
        "current_angle": result.current_angle,
        "relative_rotation": result.relative_rotation,
        "direction": result.direction,
        "confidence": result.confidence,
    }
    path.write_text(json.dumps(data, indent=2))
