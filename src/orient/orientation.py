"""Orientation estimation algorithms — each returns (center, axis_angle_0_180, confidence)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .geometry import EPSILON, normalize_angle_180


class OrientationEstimator(ABC):
    @abstractmethod
    def estimate(self, contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
        """Return (center, axis_angle in [0,180), confidence in [0,1])."""


class MomentsEstimator(OrientationEstimator):
    def estimate(self, contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
        m = cv2.moments(contour)
        if abs(m["m00"]) <= EPSILON:
            return PCAEstimator().estimate(contour)

        cx = int(round(m["m10"] / m["m00"]))
        cy = int(round(m["m01"] / m["m00"]))
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]

        angle = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
        common = math.sqrt(4.0 * mu11**2 + (mu20 - mu02)**2)
        major = (mu20 + mu02 + common) / 2.0
        minor = (mu20 + mu02 - common) / 2.0

        return (cx, cy), normalize_angle_180(angle), _confidence(major, minor)


class PCAEstimator(OrientationEstimator):
    def estimate(self, contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
        pts = contour.reshape(-1, 2).astype(np.float32)
        mean, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)
        cx, cy = int(mean[0][0]), int(mean[0][1])
        angle = math.degrees(math.atan2(eigvecs[0][1], eigvecs[0][0]))
        vals = eigvals.flatten()
        conf = _confidence(float(vals[0]), float(vals[1])) if len(vals) >= 2 else 0.0
        return (cx, cy), normalize_angle_180(angle), conf


class MinAreaRectEstimator(OrientationEstimator):
    def estimate(self, contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
        rect = cv2.minAreaRect(contour)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        w, h = rect[1]
        box = cv2.boxPoints(rect).astype(np.float32)

        longest_edge = None
        longest_len = -1.0
        for i in range(4):
            edge = box[(i + 1) % 4] - box[i]
            length = float(np.linalg.norm(edge))
            if length > longest_len:
                longest_len = length
                longest_edge = edge

        if longest_edge is None:
            return (cx, cy), 0.0, 0.0

        angle = math.degrees(math.atan2(float(longest_edge[1]), float(longest_edge[0])))
        conf = _confidence(max(w, h), min(w, h))
        return (cx, cy), normalize_angle_180(angle), conf


class EllipseEstimator(OrientationEstimator):
    def estimate(self, contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
        if len(contour) < 5:
            return PCAEstimator().estimate(contour)
        ellipse = cv2.fitEllipse(contour)
        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
        w, h = ellipse[1]
        angle = ellipse[2]
        if w > h:
            angle = normalize_angle_180(angle + 90.0)
        else:
            angle = normalize_angle_180(angle)
        conf = _confidence(max(w, h), min(w, h))
        return (cx, cy), angle, conf


class ConvexHullEstimator(OrientationEstimator):
    def estimate(self, contour: np.ndarray) -> tuple[tuple[int, int], float, float]:
        hull = cv2.convexHull(contour)
        return PCAEstimator().estimate(hull)


def _confidence(major: float, minor: float) -> float:
    if major <= EPSILON:
        return 0.0
    return max(0.0, min(1.0, (major - minor) / major))


def create_estimator(method: str) -> OrientationEstimator:
    estimators: dict[str, type[OrientationEstimator]] = {
        "moments": MomentsEstimator,
        "pca": PCAEstimator,
        "box": MinAreaRectEstimator,
        "ellipse": EllipseEstimator,
        "hull": ConvexHullEstimator,
    }
    cls = estimators.get(method)
    if cls is None:
        raise ValueError(f"Unknown orientation method: {method!r}. Choose from: {list(estimators)}")
    return cls()
