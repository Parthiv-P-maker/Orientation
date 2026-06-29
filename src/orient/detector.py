"""Object detection: segmentation + contour extraction + orientation estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .orientation import OrientationEstimator, create_estimator
from .segmentation import Segmenter, create_segmenter


@dataclass
class Detection:
    object_id: int
    area: float
    center: tuple[int, int]
    contour: np.ndarray
    box: np.ndarray
    axis_angle: float
    confidence: float
    width: float
    height: float


class ObjectDetector:
    def __init__(self, segmenter: Segmenter, estimator: OrientationEstimator, min_area: float = 1000.0, max_objects: int | None = None) -> None:
        self.segmenter = segmenter
        self.estimator = estimator
        self.min_area = min_area
        self.max_objects = max_objects

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        mask = self.segmenter.segment(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            rect = cv2.minAreaRect(contour)
            box = np.int32(np.round(cv2.boxPoints(rect)))
            w, h = rect[1]
            center, axis_angle, confidence = self.estimator.estimate(contour)

            detections.append(Detection(
                object_id=0,
                area=area,
                center=center,
                contour=contour,
                box=box,
                axis_angle=axis_angle,
                confidence=confidence,
                width=w,
                height=h,
            ))

        detections.sort(key=lambda d: d.area, reverse=True)
        if self.max_objects is not None:
            detections = detections[:self.max_objects]
        for i, det in enumerate(detections, 1):
            det.object_id = i

        return detections, mask


def create_detector(
    mask_mode: str = "color",
    method: str = "moments",
    min_area: float = 1000.0,
    max_objects: int | None = None,
    **seg_kwargs,
) -> ObjectDetector:
    segmenter = create_segmenter(mask_mode, **seg_kwargs)
    estimator = create_estimator(method)
    return ObjectDetector(segmenter, estimator, min_area, max_objects)
