"""Tests for geometry utilities."""

import pytest
from orient.geometry import normalize_angle_180, normalize_signed, relative_angle, rotation_direction


class TestNormalizeAngle180:
    def test_zero(self):
        assert normalize_angle_180(0.0) == 0.0

    def test_positive(self):
        assert normalize_angle_180(45.0) == 45.0

    def test_wrap(self):
        assert normalize_angle_180(200.0) == pytest.approx(20.0)

    def test_negative(self):
        assert normalize_angle_180(-10.0) == pytest.approx(170.0)


class TestNormalizeSigned:
    def test_zero(self):
        assert normalize_signed(0.0) == 0.0

    def test_positive(self):
        assert normalize_signed(90.0) == 90.0

    def test_wrap_positive(self):
        assert normalize_signed(270.0) == pytest.approx(-90.0)

    def test_negative(self):
        assert normalize_signed(-90.0) == pytest.approx(-90.0)


class TestRelativeAngle:
    def test_same(self):
        assert relative_angle(45.0, 45.0) == pytest.approx(0.0)

    def test_positive_rotation(self):
        assert relative_angle(90.0, 45.0) == pytest.approx(45.0)

    def test_negative_rotation(self):
        assert relative_angle(10.0, 50.0) == pytest.approx(-40.0)


class TestRotationDirection:
    def test_cw(self):
        assert rotation_direction(-10.0) == "Clockwise"

    def test_ccw(self):
        assert rotation_direction(10.0) == "Counter-clockwise"

    def test_none(self):
        assert rotation_direction(0.0) == "None"
