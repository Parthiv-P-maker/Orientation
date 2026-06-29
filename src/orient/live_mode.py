"""Live camera mode — press SPACE to set reference, R to reset."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .detector import ObjectDetector
from .logger import DataLogger, LogEntry
from .reference_manager import ReferenceManager
from .tracker import CentroidTracker
from .visualization import Visualizer


def run_live_mode(
    camera_index: int,
    detector: ObjectDetector,
    output_dir: Path,
    camera_width: int = 1280,
    camera_height: int = 720,
    output_video: str | None = None,
    save_csv: bool = False,
    save_json: bool = False,
    decimals: int = 2,
    max_disappeared: int = 15,
    max_distance: float = 80.0,
    show_debug: bool = False,
    inspection_enabled: bool = False,
    target_angle: float = 0.0,
    tolerance: float = 2.0,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    ref_mgr = ReferenceManager()
    tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
    viz = Visualizer(decimals=decimals)
    logger = DataLogger()
    writer: cv2.VideoWriter | None = None
    frame_idx = 0

    print("Live Mode — SPACE: set reference | R: reset reference | Q/ESC: quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t0 = time.time()

            detections, mask = detector.detect(frame)
            measurements = [ref_mgr.calculate_relative(d.axis_angle, d.confidence) for d in detections]

            tracked = tracker.update(
                [d.center for d in detections],
                [d.axis_angle for d in detections],
                [d.confidence for d in detections],
            )

            inspection_results: list[str] | None = None
            if inspection_enabled and ref_mgr.is_set:
                inspection_results = []
                for m in measurements:
                    diff = abs(m.relative_angle - target_angle)
                    inspection_results.append("PASS" if diff <= tolerance else "FAIL")

            annotated = viz.annotate(
                frame, detections, measurements,
                reference_set=ref_mgr.is_set,
                tracking_ids=[t.object_id for t in tracked] if tracked else None,
                angular_velocities=[t.angular_velocity for t in tracked] if tracked else None,
                inspection_results=inspection_results,
            )

            if not ref_mgr.is_set:
                cv2.putText(annotated, "Press SPACE to set reference", (10, annotated.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            for i, (det, meas) in enumerate(zip(detections, measurements)):
                av = tracked[i].angular_velocity if i < len(tracked) else 0.0
                logger.log(LogEntry(
                    frame=frame_idx,
                    timestamp=t0,
                    object_id=tracked[i].object_id if i < len(tracked) else det.object_id,
                    reference_angle=meas.reference_angle,
                    current_angle=meas.current_angle,
                    relative_angle=meas.relative_angle,
                    angular_velocity=av,
                    confidence=meas.confidence,
                    tracking_status="tracked" if ref_mgr.is_set else "no_ref",
                    center_x=det.center[0],
                    center_y=det.center[1],
                    bbox_w=det.width,
                    bbox_h=det.height,
                ))

            if writer is None and output_video:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = Path(output_video)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("Live Orientation", annotated)
            if show_debug:
                cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" ") and detections:
                d = detections[0]
                ref_mgr.set_reference(d.contour, d.center, d.axis_angle, frame, d.box, d.width, d.height)
                print(f"Reference SET: {d.axis_angle:.2f} deg")
            elif key == ord("r"):
                ref_mgr.clear_reference()
                tracker.reset()
                print("Reference RESET")
            elif key == ord("s"):
                ref_mgr.save_reference(output_dir / "reference")
                print("Reference saved")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        logger.save_csv(output_dir / "live_log.csv")
    if save_json:
        logger.save_json(output_dir / "live_log.json")
