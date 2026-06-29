"""Tests for CentroidTracker."""

from orient.tracker import CentroidTracker


class TestCentroidTracker:
    def test_register_new(self):
        t = CentroidTracker()
        objs = t.update([(100, 100)], [45.0], [0.9])
        assert len(objs) == 1
        assert objs[0].object_id == 1

    def test_track_movement(self):
        t = CentroidTracker()
        t.update([(100, 100)], [45.0], [0.9])
        objs = t.update([(105, 102)], [47.0], [0.9])
        assert len(objs) == 1
        assert objs[0].object_id == 1

    def test_disappear(self):
        t = CentroidTracker(max_disappeared=2)
        t.update([(100, 100)], [45.0], [0.9])
        t.update([], [], [])
        t.update([], [], [])
        objs = t.update([], [], [])
        assert len(objs) == 0

    def test_multiple_objects(self):
        t = CentroidTracker()
        objs = t.update([(50, 50), (200, 200)], [10.0, 80.0], [0.8, 0.7])
        assert len(objs) == 2
