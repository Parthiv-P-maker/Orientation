"""Tests for the ReferenceManager."""

import numpy as np
import pytest

from orient.reference_manager import ReferenceManager


def _dummy_contour():
    return np.array([[[10, 10]], [[100, 10]], [[100, 50]], [[10, 50]]], dtype=np.int32)


class TestReferenceManager:
    def test_initially_unset(self):
        rm = ReferenceManager()
        assert not rm.is_set

    def test_set_and_clear(self):
        rm = ReferenceManager()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        box = np.array([[10, 10], [100, 10], [100, 50], [10, 50]], dtype=np.int32)
        rm.set_reference(_dummy_contour(), (55, 30), 45.0, img, box)
        assert rm.is_set
        rm.clear_reference()
        assert not rm.is_set

    def test_relative_measurement(self):
        rm = ReferenceManager()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        box = np.array([[10, 10], [100, 10], [100, 50], [10, 50]], dtype=np.int32)
        rm.set_reference(_dummy_contour(), (55, 30), 30.0, img, box)
        meas = rm.calculate_relative(75.0, 0.9)
        assert meas.relative_angle == pytest.approx(45.0)
        assert meas.direction == "Counter-clockwise"

    def test_no_reference(self):
        rm = ReferenceManager()
        meas = rm.calculate_relative(45.0, 0.8)
        assert meas.direction == "No reference"
