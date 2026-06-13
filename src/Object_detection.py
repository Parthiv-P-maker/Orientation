from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".wmv"}


@dataclass
class Detection:
    object_id: int
    area: float
    center: tuple[int, int]
    box: np.ndarray
    axis_angle: float
    display_angle: float
    axis_length: int


def default_source() -> str:
    for candidate in (
        PROJECT_ROOT / "Images" / "test.jpg",
        PROJECT_ROOT / "Images" / "pinkcup.jpeg",PROJECT_ROOT / "Images" / "testvid1.mp4",
    ):
        if candidate.exists():
            return str(candidate)
    return str(PROJECT_ROOT / "Images")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect colored/dark/bright objects and display their orientation angle."
    )
    parser.add_argument(
        "--source",
        default=default_source(),
        help="Image/video path or numeric camera index. Defaults to a sample image.",
    )
    parser.add_argument(
        "--webcam",
        nargs="?",
        const=0,
        type=int,
        help="Use a webcam feed. Optionally pass the camera index, for example --webcam 1.",
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
        choices=("pca", "box"),
        default="pca",
        help="Orientation method: PCA major axis or min-area-rectangle long side.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1000.0,
        help="Ignore contours smaller than this pixel area.",
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


def build_mask(frame: np.ndarray, mode: str) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    if mode == "color":
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 45, 45], dtype=np.uint8)
        upper = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        if mode == "dark":
            _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def normalize_axis_angle(angle_degrees: float) -> float:
    return angle_degrees % 180.0


def angle_from_horizontal(axis_angle: float) -> float:
    angle = normalize_axis_angle(axis_angle)
    if angle > 90.0:
        return 180.0 - angle
    return angle


def pca_axis_angle(contour: np.ndarray) -> tuple[tuple[int, int], float]:
    points = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    center = (int(mean[0][0]), int(mean[0][1]))
    vector_x, vector_y = eigenvectors[0]
    axis_angle = normalize_axis_angle(math.degrees(math.atan2(vector_y, vector_x)))
    return center, axis_angle


def box_axis_angle(box: np.ndarray) -> float:
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
        return 0.0

    return normalize_axis_angle(
        math.degrees(math.atan2(float(longest_edge[1]), float(longest_edge[0])))
    )


def detect_objects(
    frame: np.ndarray,
    min_area: float,
    mask_mode: str,
    method: str,
) -> tuple[list[Detection], np.ndarray]:
    mask = build_mask(frame, mask_mode)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        if method == "pca" and len(contour) >= 2:
            center, axis_angle = pca_axis_angle(contour)
        else:
            center = (int(rect[0][0]), int(rect[0][1]))
            axis_angle = box_axis_angle(box.astype(np.float32))

        detections.append(
            Detection(
                object_id=0,
                area=area,
                center=center,
                box=box,
                axis_angle=axis_angle,
                display_angle=angle_from_horizontal(axis_angle),
                axis_length=axis_length,
            )
        )

    detections.sort(key=lambda item: item.area, reverse=True)
    for object_id, detection in enumerate(detections, start=1):
        detection.object_id = object_id

    return detections, mask


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()

    for detection in detections:
        center_x, center_y = detection.center
        theta = math.radians(detection.axis_angle)
        dx = int(math.cos(theta) * detection.axis_length)
        dy = int(math.sin(theta) * detection.axis_length)
        start = (center_x - dx, center_y - dy)
        end = (center_x + dx, center_y + dy)

        cv2.drawContours(annotated, [detection.box], 0, (0, 255, 0), 2)
        cv2.circle(annotated, detection.center, 5, (0, 0, 255), -1)
        cv2.line(annotated, start, end, (255, 0, 0), 2)

        label = f"Obj {detection.object_id}: {detection.display_angle:.1f} deg"
        label_x = min(center_x + 12, max(0, annotated.shape[1] - 190))
        label_y = max(24, center_y - 12)
        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def print_summary(source_name: str, detections: list[Detection]) -> None:
    print(f"Source: {source_name}")
    print(f"Detected Objects: {len(detections)}")

    for detection in detections:
        print(
            f"Object {detection.object_id}: "
            f"Area={detection.area:.0f}, "
            f"Center={detection.center}, "
            f"Angle={detection.display_angle:.2f} deg"
        )


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


def run_image(args: argparse.Namespace, source_path: Path) -> None:
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {source_path}")

    detections, mask = detect_objects(
        image,
        min_area=args.min_area,
        mask_mode=args.mask_mode,
        method=args.method,
    )
    annotated = draw_detections(image, detections)
    print_summary(str(source_path), detections)

    if args.output:
        save_image(args.output, annotated)

    if not args.no_display:
        display_image("Orientation Detection", annotated, mask, args.show_mask)


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
            )
            annotated = draw_detections(frame, detections)

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


def source_is_camera(source: str) -> bool:
    return source.strip().isdigit()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.webcam is not None:
        run_stream(args, args.webcam)
        return

    source = str(args.source)
    if source_is_camera(source):
        run_stream(args, int(source))
        return

    source_path = Path(source)
    suffix = source_path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        run_image(args, source_path)
    elif suffix in VIDEO_EXTENSIONS:
        run_stream(args, str(source_path))
    else:
        raise SystemExit(
            "Unsupported source. Use an image, video, numeric camera index, or --webcam."
        )


if __name__ == "__main__":
    main()
