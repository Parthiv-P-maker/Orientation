"""Object segmentation from frames using multiple strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Segmenter(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a binary mask of detected objects."""


class HSVSegmenter(Segmenter):
    def __init__(
        self,
        hue_min: int = 0,
        hue_max: int = 179,
        sat_min: int = 55,
        value_min: int = 45,
    ) -> None:
        self.hue_min = max(0, min(179, hue_min))
        self.hue_max = max(0, min(179, hue_max))
        self.sat_min = max(0, min(255, sat_min))
        self.value_min = max(0, min(255, value_min))

    def segment(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if self.hue_min <= self.hue_max:
            mask = cv2.inRange(
                hsv,
                np.array([self.hue_min, self.sat_min, self.value_min], dtype=np.uint8),
                np.array([self.hue_max, 255, 255], dtype=np.uint8),
            )
        else:
            lower = cv2.inRange(
                hsv,
                np.array([0, self.sat_min, self.value_min], dtype=np.uint8),
                np.array([self.hue_max, 255, 255], dtype=np.uint8),
            )
            upper = cv2.inRange(
                hsv,
                np.array([self.hue_min, self.sat_min, self.value_min], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(lower, upper)

        return _clean_mask(mask)


class ThresholdSegmenter(Segmenter):
    def __init__(self, invert: bool = False) -> None:
        self.invert = invert

    def segment(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        flags = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, flags + cv2.THRESH_OTSU)
        return _clean_mask(mask)


class AdaptiveSegmenter(Segmenter):
    def __init__(self, block_size: int = 11, c: int = 2) -> None:
        self.block_size = block_size | 1
        self.c = c

    def segment(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.c,
        )
        return _clean_mask(mask)


class EdgeSegmenter(Segmenter):
    def __init__(self, low: int = 50, high: int = 150) -> None:
        self.low = low
        self.high = high

    def segment(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low, self.high)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return _clean_mask(mask)


class BackgroundSubtractorSegmenter(Segmenter):
    def __init__(self) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False,
        )

    def segment(self, frame: np.ndarray) -> np.ndarray:
        mask = self.subtractor.apply(frame)
        return _clean_mask(mask)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def create_segmenter(mode: str, **kwargs) -> Segmenter:
    factories = {
        "color": lambda: HSVSegmenter(**{k: kwargs[k] for k in ("hue_min", "hue_max", "sat_min", "value_min") if k in kwargs}),
        "hsv": lambda: HSVSegmenter(**{k: kwargs[k] for k in ("hue_min", "hue_max", "sat_min", "value_min") if k in kwargs}),
        "dark": lambda: ThresholdSegmenter(invert=True),
        "bright": lambda: ThresholdSegmenter(invert=False),
        "adaptive": lambda: AdaptiveSegmenter(),
        "edge": lambda: EdgeSegmenter(),
        "background": lambda: BackgroundSubtractorSegmenter(),
    }
    factory = factories.get(mode)
    if factory is None:
        raise ValueError(f"Unknown segmentation mode: {mode!r}. Choose from: {list(factories)}")
    return factory()
