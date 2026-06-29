"""Image vs Image comparison mode."""

from __future__ import annotations

from pathlib import Path

import cv2

from .comparison import annotate_comparison, compare_images, save_comparison_report
from .detector import ObjectDetector


def run_image_mode(
    reference_path: str,
    comparison_path: str,
    detector: ObjectDetector,
    output_dir: Path,
    decimals: int = 2,
    no_display: bool = False,
) -> None:
    ref_img = cv2.imread(reference_path)
    cur_img = cv2.imread(comparison_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot read reference image: {reference_path}")
    if cur_img is None:
        raise FileNotFoundError(f"Cannot read comparison image: {comparison_path}")

    result = compare_images(ref_img, cur_img, detector)
    if result is None:
        print("No objects detected in one or both images.")
        return

    print(f"Reference Angle: {result.reference_angle:.{decimals}f} deg")
    print(f"Current Angle:   {result.current_angle:.{decimals}f} deg")
    print(f"Relative Rotation: {result.relative_rotation:+.{decimals}f} deg")
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence * 100:.0f}%")

    canvas = annotate_comparison(ref_img, cur_img, result, decimals)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "comparison.png"), canvas)
    save_comparison_report(result, output_dir / "comparison.json")
    print(f"Saved to {output_dir}")

    if not no_display:
        cv2.imshow("Image Comparison", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
