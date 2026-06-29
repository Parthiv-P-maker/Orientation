"""Angle normalization and geometric utilities."""

from __future__ import annotations

import math

EPSILON = 1e-7


def normalize_angle_180(degrees: float) -> float:
    """Normalize angle to [0, 180) — undirected axis range."""
    return degrees % 180.0


def normalize_angle_360(degrees: float) -> float:
    """Normalize angle to [0, 360)."""
    return degrees % 360.0


def normalize_signed(degrees: float) -> float:
    """Normalize angle to (-180, 180]."""
    a = degrees % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def relative_angle(current: float, reference: float) -> float:
    """Compute signed relative rotation from reference to current, in (-180, 180]."""
    return normalize_signed(current - reference)


def rotation_direction(rel_angle: float) -> str:
    if abs(rel_angle) < EPSILON:
        return "None"
    return "Counter-clockwise" if rel_angle > 0 else "Clockwise"


def angular_velocity(angle_delta: float, time_delta: float) -> float:
    if time_delta <= EPSILON:
        return 0.0
    return angle_delta / time_delta


def point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
