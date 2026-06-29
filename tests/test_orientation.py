"""Tests for orientation estimators."""

import cv2
import numpy as np
import pytest

from orient.orientation import MomentsEstimator, PCAEstimator, MinAreaRectEstimator, EllipseEstimator


def _make_rect_contour(cx: int, cy: int, w: int, h: int, angle_deg: float) -> np.ndarray:
    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect)
    return np.int32(box).reshape(-1, 1, 2)


class TestMomentsEstimator:
    def test_horizontal_rect(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 80), (150, 120), 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        center, angle, conf = MomentsEstimator().estimate(contours[0])
        assert abs(angle) < 5.0 or abs(angle - 180.0) < 5.0

    def test_confidence_elongated(self):
        img = np.zeros((200, 400), dtype=np.uint8)
        cv2.rectangle(img, (50, 90), (350, 110), 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, _, conf = MomentsEstimator().estimate(contours[0])
        assert conf > 0.5


class TestPCAEstimator:
    def test_basic(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 80), (150, 120), 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        center, angle, conf = PCAEstimator().estimate(contours[0])
        assert 0.0 <= angle < 180.0
