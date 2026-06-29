"""Generate a synthetic test image with a rectangle at a known angle.

Useful for validating orientation estimators against ground truth.

Usage:
    python tools/generate_test_image.py --angle 30 --output samples/test_rect_30deg.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_rect_image(
    size: int = 500,
    rect_size: tuple[int, int] = (200, 80),
    angle_deg: float = 30.0,
) -> np.ndarray:
    """Render a filled black rectangle rotated by ``angle_deg`` on white."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    rect = ((size // 2, size // 2), rect_size, angle_deg)
    box = np.int32(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], 0, (0, 0, 0), -1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a test rectangle image.")
    parser.add_argument("--angle", type=float, default=30.0, help="Rotation angle in degrees.")
    parser.add_argument("--output", default="samples/test_rect.png", help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Display the generated image.")
    args = parser.parse_args()

    img = generate_rect_image(angle_deg=args.angle)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved {output_path} (rectangle at {args.angle} deg)")

    if args.show:
        cv2.imshow("Generated", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
