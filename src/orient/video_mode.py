"""Video analysis mode — first frame is the reference."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .detector import ObjectDetector
from .logger import DataLogger, LogEntry
from .reference_manager import ReferenceManager
from .tracker import CentroidTracker
from .visualization import Visualizer


def run_video_mode(
    source: str,
    detector: ObjectDetector,
    output_dir: Path,
    output_video: str | None = None,
    no_display: bool = False,
    save_csv: bool = False,
    save_json: bool = False,
    decimals: int = 2,
    max_disappeared: int = 15,
    max_distance: float = 80.0,
    show_debug: bool = False,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    ref_mgr = ReferenceManager()
    tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
    viz = Visualizer(decimals=decimals)
    logger = DataLogger()
    writer: cv2.VideoWriter | None = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t0 = time.time()

            detections, mask = detector.detect(frame)

            if not ref_mgr.is_set and detections:
                d = detections[0]
                ref_mgr.set_reference(d.contour, d.center, d.axis_angle, frame, d.box, d.width, d.height)
                print(f"Reference set from frame 1: {d.axis_angle:.2f} deg")

            measurements = [ref_mgr.calculate_relative(d.axis_angle, d.confidence) for d in detections]

            tracked = tracker.update(
                [d.center for d in detections],
                [d.axis_angle for d in detections],
                [d.confidence for d in detections],
            )

            annotated = viz.annotate(
                frame, detections, measurements,
                reference_set=ref_mgr.is_set,
                tracking_ids=[t.object_id for t in tracked] if tracked else None,
                angular_velocities=[t.angular_velocity for t in tracked] if tracked else None,
            )

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
                    tracking_status="tracked",
                    center_x=det.center[0],
                    center_y=det.center[1],
                    bbox_w=det.width,
                    bbox_h=det.height,
                ))

            if writer is None and output_video:
                h, w = annotated.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = Path(output_video)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            if writer is not None:
                writer.write(annotated)

            if frame_idx == 1 or frame_idx % 30 == 0:
                if measurements:
                    m = measurements[0]
                    print(f"Frame {frame_idx}: Rel={m.relative_angle:+.1f}deg ({m.direction})")

            if not no_display:
                cv2.imshow("Video Analysis", annotated)
                if show_debug:
                    cv2.imshow("Mask", mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved video: {output_video}")
        if not no_display:
            cv2.destroyAllWindows()

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        logger.save_csv(output_dir / "video_log.csv")
        print(f"Saved CSV: {output_dir / 'video_log.csv'}")
    if save_json:
        logger.save_json(output_dir / "video_log.json")
        print(f"Saved JSON: {output_dir / 'video_log.json'}")

    summary = logger.summary()
    if summary:
        print(f"Summary: {summary}")
