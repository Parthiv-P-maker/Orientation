"""Object tracking across frames."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .geometry import angular_velocity, point_distance, relative_angle


@dataclass
class TrackedObject:
    object_id: int
    centroid: tuple[int, int]
    angle: float
    confidence: float
    disappeared: int = 0
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    angle_history: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    angular_velocity: float = 0.0


class CentroidTracker:
    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id = 1
        self._objects: OrderedDict[int, TrackedObject] = OrderedDict()

    @property
    def objects(self) -> dict[int, TrackedObject]:
        return dict(self._objects)

    def update(
        self,
        centroids: list[tuple[int, int]],
        angles: list[float],
        confidences: list[float],
    ) -> list[TrackedObject]:
        now = time.time()

        if len(centroids) == 0:
            for oid in list(self._objects):
                self._objects[oid].disappeared += 1
                if self._objects[oid].disappeared > self.max_disappeared:
                    del self._objects[oid]
            return list(self._objects.values())

        if len(self._objects) == 0:
            for c, a, conf in zip(centroids, angles, confidences):
                self._register(c, a, conf, now)
            return list(self._objects.values())

        obj_ids = list(self._objects.keys())
        obj_centroids = [self._objects[oid].centroid for oid in obj_ids]

        dist_matrix = np.zeros((len(obj_ids), len(centroids)), dtype=np.float64)
        for i, oc in enumerate(obj_centroids):
            for j, nc in enumerate(centroids):
                dist_matrix[i, j] = point_distance(oc, nc)

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        matches: list[tuple[int, int]] = []

        flat_indices = dist_matrix.argsort(axis=None)
        for idx in flat_indices:
            r, c = divmod(int(idx), len(centroids))
            if r in used_rows or c in used_cols:
                continue
            if dist_matrix[r, c] > self.max_distance:
                continue
            matches.append((r, c))
            used_rows.add(r)
            used_cols.add(c)

        for r, c in matches:
            oid = obj_ids[r]
            obj = self._objects[oid]
            prev_angle = obj.angle
            prev_time = obj.timestamps[-1] if obj.timestamps else now

            obj.centroid = centroids[c]
            obj.angle = angles[c]
            obj.confidence = confidences[c]
            obj.disappeared = 0
            obj.trajectory.append(centroids[c])
            obj.angle_history.append(angles[c])
            obj.timestamps.append(now)
            obj.angular_velocity = angular_velocity(
                relative_angle(angles[c], prev_angle), now - prev_time,
            )

        for r in range(len(obj_ids)):
            if r not in used_rows:
                oid = obj_ids[r]
                self._objects[oid].disappeared += 1
                if self._objects[oid].disappeared > self.max_disappeared:
                    del self._objects[oid]

        for c in range(len(centroids)):
            if c not in used_cols:
                self._register(centroids[c], angles[c], confidences[c], now)

        return list(self._objects.values())

    def _register(self, centroid: tuple[int, int], angle: float, confidence: float, ts: float) -> None:
        obj = TrackedObject(
            object_id=self._next_id,
            centroid=centroid,
            angle=angle,
            confidence=confidence,
            trajectory=[centroid],
            angle_history=[angle],
            timestamps=[ts],
        )
        self._objects[self._next_id] = obj
        self._next_id += 1

    def reset(self) -> None:
        self._objects.clear()
        self._next_id = 1
