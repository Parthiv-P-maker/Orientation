from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-7


@dataclass
class Detection:
    object_id: int
    area: float
    center: tuple[int, int]
    box: np.ndarray
    axis_angle: float
    signed_angle: float
    acute_angle: float
    vertical_axis_angle: float
    vertical_signed_angle: float
    vertical_acute_angle: float
    confidence: float
    width: float
    height: float
    axis_length: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect objects from a webcam feed and display their orientation angles."
    )
    parser.add_argument(
        "--webcam",
        nargs="?",
        const=0,
        default=0,
        type=int,
        help="Use a webcam feed. Optionally pass the camera index, for example --webcam 1. Defaults to 0.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the annotated output image or video.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("color", "dark", "bright"),
        default="color",
        help="Segmentation mode. 'color' works well for saturated objects.",
    )
    parser.add_argument(
        "--method",
        choices=("moments", "pca", "box"),
        default="moments",
        help="Orientation method: image moments, PCA major axis, or min-area-rectangle long side.",
    )
    parser.add_argument(
        "--angle-format",
        choices=("acute", "axis", "signed"),
        default="acute",
        help=(
            "Angle shown on the image: acute=0..90 from horizontal, "
            "axis=0..180, signed=-90..90."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places shown for angles.",
    )
    parser.add_argument(
        "--hue-min",
        type=int,
        default=0,
        help="Minimum HSV hue for color segmentation, 0 to 179.",
    )
    parser.add_argument(
        "--hue-max",
        type=int,
        default=179,
        help="Maximum HSV hue for color segmentation, 0 to 179.",
    )
    parser.add_argument(
        "--sat-min",
        type=int,
        default=55,
        help="Minimum HSV saturation for color segmentation.",
    )
    parser.add_argument(
        "--value-min",
        type=int,
        default=45,
        help="Minimum HSV value/brightness for color segmentation.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1000.0,
        help="Ignore contours smaller than this pixel area.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        help="Only keep the largest N detected objects.",
    )
    parser.add_argument(
        "--largest-only",
        action="store_true",
        help="Shortcut for --max-objects 1.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Only print the best detection by confidence and highlight it in the output.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the binary mask window while processing.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without OpenCV display windows. Useful for saving output or testing.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested webcam capture height.",
    )
    return parser


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def build_color_mask(
    hsv: np.ndarray,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    value_min: int,
) -> np.ndarray:
    hue_min = clamp_int(hue_min, 0, 179)
    hue_max = clamp_int(hue_max, 0, 179)
    sat_min = clamp_int(sat_min, 0, 255)
    value_min = clamp_int(value_min, 0, 255)

    if hue_min <= hue_max:
        return cv2.inRange(
            hsv,
            np.array([hue_min, sat_min, value_min], dtype=np.uint8),
            np.array([hue_max, 255, 255], dtype=np.uint8),
        )

    lower_wrap = cv2.inRange(
        hsv,
        np.array([0, sat_min, value_min], dtype=np.uint8),
        np.array([hue_max, 255, 255], dtype=np.uint8),
    )
    upper_wrap = cv2.inRange(
        hsv,
        np.array([hue_min, sat_min, value_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    return cv2.bitwise_or(lower_wrap, upper_wrap)


def build_mask(
    frame: np.ndarray,
    mode: str,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    value_min: int,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    if mode == "color":
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = build_color_mask(hsv, hue_min, hue_max, sat_min, value_min)
    else:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        if mode == "dark":
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def normalize_axis_angle(angle_degrees: float) -> float:
    return angle_degrees % 180.0


def signed_axis_angle(axis_angle: float) -> float:
    angle = normalize_axis_angle(axis_angle)
    if angle > 90.0:
        angle -= 180.0
    return angle


def acute_axis_angle(axis_angle: float) -> float:
    return abs(signed_axis_angle(axis_angle))


def vertical_axis_angle(axis_angle: float) -> float:
    return normalize_axis_angle(axis_angle - 90.0)


def signed_vertical_axis_angle(axis_angle: float) -> float:
    angle = vertical_axis_angle(axis_angle)
    if angle > 90.0:
        angle -= 180.0
    return angle


def acute_vertical_axis_angle(axis_angle: float) -> float:
    return abs(signed_vertical_axis_angle(axis_angle))


def orientation_confidence(major_value: float, minor_value: float) -> float:
    if major_value <= EPSILON:
        return 0.0
    confidence = (major_value - minor_value) / major_value
    return max(0.0, min(1.0, confidence))


def moments_axis_angle(contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) <= EPSILON:
        return pca_axis_angle(contour)

    center = (
        int(round(moments["m10"] / moments["m00"])),
        int(round(moments["m01"] / moments["m00"])),
    )
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]

    angle = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
    common = math.sqrt(4.0 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02))
    major_value = (mu20 + mu02 + common) / 2.0
    minor_value = (mu20 + mu02 - common) / 2.0

    return (
        center,
        normalize_axis_angle(angle),
        orientation_confidence(major_value, minor_value),
    )


def pca_axis_angle(contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
    points = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    center = (int(mean[0][0]), int(mean[0][1]))
    vector_x, vector_y = eigenvectors[0]
    axis_angle = normalize_axis_angle(math.degrees(math.atan2(vector_y, vector_x)))
    values = eigenvalues.flatten()
    confidence = (
        orientation_confidence(float(values[0]), float(values[1]))
        if len(values) >= 2
        else 0.0
    )
    return center, axis_angle, confidence


def box_axis_angle(box: np.ndarray, width: float, height: float) -> tuple[float, float]:
    longest_edge = None
    longest_length = -1.0

    for index in range(4):
        start = box[index]
        end = box[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length > longest_length:
            longest_length = length
            longest_edge = edge

    if longest_edge is None:
        return 0.0, 0.0

    long_side = max(width, height)
    short_side = min(width, height)
    confidence = 0.0
    if long_side > EPSILON:
        confidence = orientation_confidence(long_side, short_side)

    return (
        normalize_axis_angle(
            math.degrees(math.atan2(float(longest_edge[1]), float(longest_edge[0])))
        ),
        confidence,
    )


def detect_objects(
    frame: np.ndarray,
    min_area: float,
    mask_mode: str,
    method: str,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    value_min: int,
    max_objects: int | None,
) -> tuple[list[Detection], np.ndarray]:
    mask = build_mask(frame, mask_mode, hue_min, hue_max, sat_min, value_min)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(np.round(box))
        width, height = rect[1]
        axis_length = max(35, int(max(width, height) * 0.55))

        if method == "moments":
            center, axis_angle, confidence = moments_axis_angle(contour)
        elif method == "pca" and len(contour) >= 2:
            center, axis_angle, confidence = pca_axis_angle(contour)
        else:
            center = (int(rect[0][0]), int(rect[0][1]))
            axis_angle, confidence = box_axis_angle(box.astype(np.float32), width, height)

        detections.append(
            Detection(
                object_id=0,
                area=area,
                center=center,
                box=box,
                axis_angle=axis_angle,
                signed_angle=signed_axis_angle(axis_angle),
                acute_angle=acute_axis_angle(axis_angle),
                vertical_axis_angle=vertical_axis_angle(axis_angle),
                vertical_signed_angle=signed_vertical_axis_angle(axis_angle),
                vertical_acute_angle=acute_vertical_axis_angle(axis_angle),
                confidence=confidence,
                width=width,
                height=height,
                axis_length=axis_length,
            )
        )

    detections.sort(key=lambda item: item.area, reverse=True)
    if max_objects is not None:
        detections = detections[:max_objects]

    for object_id, detection in enumerate(detections, start=1):
        detection.object_id = object_id

    return detections, mask


def angle_value(detection: Detection, angle_format: str) -> float:
    if angle_format == "axis":
        return detection.axis_angle
    if angle_format == "signed":
        return detection.signed_angle
    return detection.acute_angle


def angle_name(angle_format: str) -> str:
    if angle_format == "axis":
        return "Axis"
    if angle_format == "signed":
        return "Signed"
    return "Angle"


def vertical_angle_value(detection: Detection, angle_format: str) -> float:
    if angle_format == "axis":
        return detection.vertical_axis_angle
    if angle_format == "signed":
        return detection.vertical_signed_angle
    return detection.vertical_acute_angle


def formatted_angle(detection: Detection, angle_format: str, decimals: int) -> str:
    decimals = clamp_int(decimals, 0, 4)
    value = angle_value(detection, angle_format)
    if angle_format == "signed":
        return f"{value:+.{decimals}f} deg"
    return f"{value:.{decimals}f} deg"


def formatted_vertical_angle(detection: Detection, angle_format: str, decimals: int) -> str:
    decimals = clamp_int(decimals, 0, 4)
    value = vertical_angle_value(detection, angle_format)
    if angle_format == "signed":
        return f"{value:+.{decimals}f} deg"
    return f"{value:.{decimals}f} deg"


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    padding = 5
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = clamp_int(anchor[0], padding, max(padding, image.shape[1] - text_width - padding))
    y = clamp_int(
        anchor[1],
        text_height + padding * 2,
        max(text_height + padding * 2, image.shape[0] - baseline - padding),
    )

    top_left = (x - padding, y - text_height - padding)
    bottom_right = (x + text_width + padding, y + baseline + padding)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    angle_format: str,
    decimals: int,
    best_object_id: int | None = None,
) -> np.ndarray:
    annotated = frame.copy()

    for detection in detections:
        center_x, center_y = detection.center
        theta = math.radians(detection.axis_angle)
        dx = int(math.cos(theta) * detection.axis_length)
        dy = int(math.sin(theta) * detection.axis_length)
        start = (center_x - dx, center_y - dy)
        end = (center_x + dx, center_y + dy)
        reference_length = max(40, int(detection.axis_length * 0.7))
        reference_start = (center_x - reference_length, center_y)
        reference_end = (center_x + reference_length, center_y)
        is_best = detection.object_id == best_object_id
        box_color = (0, 255, 0) if not is_best else (0, 128, 255)
        axis_color = (255, 0, 0) if not is_best else (0, 165, 255)

        cv2.drawContours(annotated, [detection.box], 0, box_color, 2)
        cv2.circle(annotated, detection.center, 5, (0, 0, 255), -1)
        cv2.line(annotated, reference_start, reference_end, (200, 200, 200), 1)
        cv2.line(annotated, start, end, axis_color, 2)

        prefix = "* " if is_best else ""
        label = (
            f"{prefix}Obj {detection.object_id}: "
            f"H={formatted_angle(detection, angle_format, decimals)} "
            f"V={formatted_vertical_angle(detection, angle_format, decimals)} "
            f"| conf {detection.confidence * 100:.0f}%"
        )
        label_x = center_x + 12
        label_y = max(24, center_y - 12)
        draw_text_with_background(annotated, label, (label_x, label_y))

    return annotated


def print_summary(
    source_name: str,
    detections: list[Detection],
    angle_format: str,
    decimals: int,
    best_only: bool = False,
) -> None:
    print(f"Source: {source_name}")
    print(f"Detected Objects: {len(detections)}")

    to_print = detections
    if best_only and detections:
        best = get_best_detection(detections)
        if best is not None:
            print("Best detection summary:")
            to_print = [best]

    for detection in to_print:
        print(
            f"Object {detection.object_id}: "
            f"Area={detection.area:.0f}, "
            f"Center={detection.center}, "
            f"H={formatted_angle(detection, angle_format, decimals)}, "
            f"V={formatted_vertical_angle(detection, angle_format, decimals)}, "
            f"Axis={detection.axis_angle:.2f} deg, "
            f"VerticalAxis={detection.vertical_axis_angle:.2f} deg, "
            f"Signed={detection.signed_angle:+.2f} deg, "
            f"VerticalSigned={detection.vertical_signed_angle:+.2f} deg, "
            f"Confidence={detection.confidence * 100:.0f}%, "
            f"Box={detection.width:.0f}x{detection.height:.0f}"
        )


def get_best_detection(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return max(detections, key=lambda item: (item.confidence, item.area))


def display_image(title: str, annotated: np.ndarray, mask: np.ndarray, show_mask: bool) -> None:
    if show_mask:
        cv2.imshow("Mask", mask)
    cv2.imshow(title, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(path: str, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save output image: {output_path}")
    print(f"Saved annotated image: {output_path}")


def selected_max_objects(args: argparse.Namespace) -> int | None:
    if args.largest_only:
        return 1
    return args.max_objects


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_name = "mp4v" if path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "XVID"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc_name),
        fps if fps > 0 else 30.0,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer


def run_stream(args: argparse.Namespace, capture_source: int | str) -> None:
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {capture_source}")

    if isinstance(capture_source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    writer = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            detections, mask = detect_objects(
                frame,
                min_area=args.min_area,
                mask_mode=args.mask_mode,
                method=args.method,
                hue_min=args.hue_min,
                hue_max=args.hue_max,
                sat_min=args.sat_min,
                value_min=args.value_min,
                max_objects=selected_max_objects(args),
            )
            best_detection = get_best_detection(detections)
            annotated = draw_detections(
                frame,
                detections,
                args.angle_format,
                args.decimals,
                best_object_id=best_detection.object_id if best_detection else None,
            )

            if writer is None and args.output:
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS)
                writer = create_video_writer(args.output, fps, (width, height))

            if writer is not None:
                writer.write(annotated)

            if frame_index == 1 or frame_index % 30 == 0:
                print(f"Frame {frame_index}: {len(detections)} objects detected")

            if not args.no_display:
                if args.show_mask:
                    cv2.imshow("Mask", mask)
                cv2.imshow("Orientation Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.no_display and writer is None and isinstance(capture_source, int):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.min_area <= 0:
        parser.error("--min-area must be greater than 0.")
    if args.max_objects is not None and args.max_objects <= 0:
        parser.error("--max-objects must be greater than 0.")
    if not 0 <= args.hue_min <= 179:
        parser.error("--hue-min must be between 0 and 179.")
    if not 0 <= args.hue_max <= 179:
        parser.error("--hue-max must be between 0 and 179.")
    if not 0 <= args.sat_min <= 255:
        parser.error("--sat-min must be between 0 and 255.")
    if not 0 <= args.value_min <= 255:
        parser.error("--value-min must be between 0 and 255.")
    args.decimals = clamp_int(args.decimals, 0, 4)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    run_stream(args, args.webcam)


if __name__ == "__main__":
    main()
