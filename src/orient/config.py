"""Configuration management with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SegmentationConfig:
    mode: str = "color"
    hue_min: int = 0
    hue_max: int = 179
    sat_min: int = 55
    value_min: int = 45
    min_area: float = 1000.0


@dataclass
class TrackingConfig:
    tracker_type: str = "centroid"
    max_disappeared: int = 15
    max_distance: float = 80.0


@dataclass
class VisualizationConfig:
    show_reference_axes: bool = True
    show_bounding_box: bool = True
    show_centroid: bool = True
    show_orientation_axis: bool = True
    show_convex_hull: bool = False
    show_rotation_arrow: bool = True
    show_labels: bool = True
    show_dashboard: bool = True
    show_debug: bool = False
    antialiased: bool = True


@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    index: int = 0


@dataclass
class InspectionConfig:
    enabled: bool = False
    target_angle: float = 0.0
    tolerance: float = 2.0


@dataclass
class AppConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    orientation_method: str = "moments"
    angle_format: str = "relative"
    decimals: int = 2
    max_objects: int | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        data = json.loads(path.read_text())
        cfg = cls()
        if "segmentation" in data:
            cfg.segmentation = SegmentationConfig(**data["segmentation"])
        if "tracking" in data:
            cfg.tracking = TrackingConfig(**data["tracking"])
        if "visualization" in data:
            cfg.visualization = VisualizationConfig(**data["visualization"])
        if "camera" in data:
            cfg.camera = CameraConfig(**data["camera"])
        if "inspection" in data:
            cfg.inspection = InspectionConfig(**data["inspection"])
        for key in ("orientation_method", "angle_format", "decimals", "max_objects"):
            if key in data:
                setattr(cfg, key, data[key])
        return cfg
