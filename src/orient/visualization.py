"""Professional visualization overlays and dashboard."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .detector import Detection
from .reference_manager import RelativeMeasurement


class Visualizer:
    def __init__(
        self,
        show_reference_axes: bool = True,
        show_bounding_box: bool = True,
        show_centroid: bool = True,
        show_orientation_axis: bool = True,
        show_convex_hull: bool = False,
        show_rotation_arrow: bool = True,
        show_labels: bool = True,
        show_dashboard: bool = True,
        decimals: int = 2,
    ) -> None:
        self.show_reference_axes = show_reference_axes
        self.show_bounding_box = show_bounding_box
        self.show_centroid = show_centroid
        self.show_orientation_axis = show_orientation_axis
        self.show_convex_hull = show_convex_hull
        self.show_rotation_arrow = show_rotation_arrow
        self.show_labels = show_labels
        self.show_dashboard = show_dashboard
        self.decimals = decimals
        self._frame_times: list[float] = []

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        measurements: list[RelativeMeasurement],
        reference_set: bool = False,
        tracking_ids: list[int] | None = None,
        angular_velocities: list[float] | None = None,
        inspection_results: list[str] | None = None,
    ) -> np.ndarray:
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 2.0]

        out = frame.copy()

        for i, (det, meas) in enumerate(zip(detections, measurements)):
            cx, cy = det.center
            axis_len = max(35, int(max(det.width, det.height) * 0.55))

            if self.show_bounding_box:
                color = (0, 255, 0)
                if inspection_results and i < len(inspection_results):
                    color = (0, 255, 0) if inspection_results[i] == "PASS" else (0, 0, 255)
                cv2.drawContours(out, [det.box], 0, color, 2, cv2.LINE_AA)

            if self.show_convex_hull:
                hull = cv2.convexHull(det.contour)
                cv2.drawContours(out, [hull], 0, (255, 255, 0), 1, cv2.LINE_AA)

            if self.show_centroid:
                cv2.circle(out, det.center, 5, (0, 0, 255), -1, cv2.LINE_AA)

            if self.show_reference_axes and reference_set:
                ref_len = max(40, int(axis_len * 0.7))
                ref_angle = math.radians(meas.reference_angle)
                rdx = int(math.cos(ref_angle) * ref_len)
                rdy = int(math.sin(ref_angle) * ref_len)
                cv2.line(out, (cx - rdx, cy - rdy), (cx + rdx, cy + rdy), (200, 200, 200), 1, cv2.LINE_AA)

            if self.show_orientation_axis:
                theta = math.radians(det.axis_angle)
                dx = int(math.cos(theta) * axis_len)
                dy = int(math.sin(theta) * axis_len)
                cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 0, 0), 2, cv2.LINE_AA)

            if self.show_rotation_arrow and reference_set and abs(meas.relative_angle) > 1.0:
                _draw_rotation_arc(out, det.center, axis_len, meas.reference_angle, meas.relative_angle)

            if self.show_labels:
                tid = tracking_ids[i] if tracking_ids and i < len(tracking_ids) else det.object_id
                av = angular_velocities[i] if angular_velocities and i < len(angular_velocities) else 0.0
                insp = ""
                if inspection_results and i < len(inspection_results):
                    insp = f" [{inspection_results[i]}]"
                label = (
                    f"ID:{tid} Rel:{meas.relative_angle:+.{self.decimals}f}deg "
                    f"({meas.direction}) Conf:{meas.confidence*100:.0f}%{insp}"
                )
                if abs(av) > 0.1:
                    label += f" w:{av:.1f}deg/s"
                draw_label(out, label, (cx + 12, max(24, cy - 12)))

        if self.show_dashboard:
            self._draw_dashboard(out, detections, measurements, reference_set)

        return out

    def _draw_dashboard(
        self,
        out: np.ndarray,
        detections: list[Detection],
        measurements: list[RelativeMeasurement],
        reference_set: bool,
    ) -> None:
        fps = len(self._frame_times) / 2.0 if len(self._frame_times) > 1 else 0.0
        lines = [
            f"FPS: {fps:.1f}",
            f"Objects: {len(detections)}",
            f"Ref: {'LOCKED' if reference_set else 'NONE'}",
        ]
        if detections and measurements:
            m = measurements[0]
            lines.append(f"Rel: {m.relative_angle:+.1f}deg")
            lines.append(f"Dir: {m.direction}")

        y = 20
        for line in lines:
            draw_label(out, line, (10, y), color=(0, 255, 255))
            y += 28


def draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw anti-aliased text with a solid background box, clamped to image bounds."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    pad = 4
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(pad, min(anchor[0], image.shape[1] - tw - pad))
    y = max(th + pad, min(anchor[1], image.shape[0] - baseline - pad))
    cv2.rectangle(image, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_rotation_arc(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    ref_angle: float,
    rel_angle: float,
) -> None:
    r = int(radius * 0.8)
    start = -ref_angle
    end = start - rel_angle
    if start > end:
        start, end = end, start
    cv2.ellipse(image, center, (r, r), 0, start, end, (0, 200, 255), 2, cv2.LINE_AA)
