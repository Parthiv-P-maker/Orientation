"""Reference frame storage and relative angle computation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import normalize_angle_180, relative_angle, rotation_direction


@dataclass
class ReferenceFrame:
    contour: np.ndarray
    centroid: tuple[int, int]
    orientation: float
    image: np.ndarray
    timestamp: float
    bounding_box: np.ndarray
    pca_vectors: np.ndarray | None = None
    moments: dict | None = None
    width: float = 0.0
    height: float = 0.0


@dataclass
class RelativeMeasurement:
    reference_angle: float
    current_angle: float
    relative_angle: float
    direction: str
    confidence: float


class ReferenceManager:
    def __init__(self) -> None:
        self._reference: ReferenceFrame | None = None

    @property
    def is_set(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> ReferenceFrame | None:
        return self._reference

    def set_reference(
        self,
        contour: np.ndarray,
        centroid: tuple[int, int],
        orientation: float,
        image: np.ndarray,
        bounding_box: np.ndarray,
        width: float = 0.0,
        height: float = 0.0,
    ) -> None:
        pts = contour.reshape(-1, 2).astype(np.float32)
        _, eigvecs, _ = cv2.PCACompute2(pts, mean=None)

        self._reference = ReferenceFrame(
            contour=contour.copy(),
            centroid=centroid,
            orientation=normalize_angle_180(orientation),
            image=image.copy(),
            timestamp=time.time(),
            bounding_box=bounding_box.copy(),
            pca_vectors=eigvecs,
            moments=cv2.moments(contour),
            width=width,
            height=height,
        )

    def clear_reference(self) -> None:
        self._reference = None

    def calculate_relative(self, current_angle: float, confidence: float) -> RelativeMeasurement:
        if self._reference is None:
            return RelativeMeasurement(
                reference_angle=0.0,
                current_angle=normalize_angle_180(current_angle),
                relative_angle=0.0,
                direction="No reference",
                confidence=confidence,
            )
        rel = relative_angle(current_angle, self._reference.orientation)
        return RelativeMeasurement(
            reference_angle=self._reference.orientation,
            current_angle=normalize_angle_180(current_angle),
            relative_angle=rel,
            direction=rotation_direction(rel),
            confidence=confidence,
        )

    def save_reference(self, directory: Path) -> None:
        if self._reference is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "reference.png"), self._reference.image)
        np.save(str(directory / "reference_contour.npy"), self._reference.contour)
        meta = {
            "centroid": list(self._reference.centroid),
            "orientation": self._reference.orientation,
            "timestamp": self._reference.timestamp,
            "width": self._reference.width,
            "height": self._reference.height,
        }
        (directory / "reference_meta.json").write_text(json.dumps(meta, indent=2))

    def load_reference(self, directory: Path) -> bool:
        img_path = directory / "reference.png"
        contour_path = directory / "reference_contour.npy"
        meta_path = directory / "reference_meta.json"
        if not all(p.exists() for p in (img_path, contour_path, meta_path)):
            return False

        image = cv2.imread(str(img_path))
        contour = np.load(str(contour_path))
        meta = json.loads(meta_path.read_text())

        self.set_reference(
            contour=contour,
            centroid=tuple(meta["centroid"]),
            orientation=meta["orientation"],
            image=image,
            bounding_box=np.int32(np.round(cv2.boxPoints(cv2.minAreaRect(contour)))),
            width=meta.get("width", 0.0),
            height=meta.get("height", 0.0),
        )
        return True
