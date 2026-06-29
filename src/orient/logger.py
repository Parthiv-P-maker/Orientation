"""Data logging to CSV and JSON."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LogEntry:
    frame: int
    timestamp: float
    object_id: int
    reference_angle: float
    current_angle: float
    relative_angle: float
    angular_velocity: float
    confidence: float
    tracking_status: str
    center_x: int
    center_y: int
    bbox_w: float
    bbox_h: float


class DataLogger:
    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def log(self, entry: LogEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[LogEntry]:
        return self._entries

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._entries:
            return
        fields = list(asdict(self._entries[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(asdict(entry))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries]
        path.write_text(json.dumps(data, indent=2))

    def summary(self) -> dict:
        if not self._entries:
            return {}
        angles = [e.relative_angle for e in self._entries]
        return {
            "total_frames": max(e.frame for e in self._entries),
            "total_entries": len(self._entries),
            "min_relative_angle": min(angles),
            "max_relative_angle": max(angles),
            "mean_relative_angle": sum(angles) / len(angles),
        }

    def clear(self) -> None:
        self._entries.clear()
