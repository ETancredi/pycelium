# tropisms/sect_field_finder.py

from __future__ import annotations
from tropisms.field_finder import FieldFinder
from core.point import MPoint
import numpy as np

class SectFieldFinder(FieldFinder):
    """
    Emits a field along a Section's line segment.
    Used for autotropism or self-avoidance.
    """

    def __init__(self, section: "Section", strength: float = 1.0, decay: float = 1.0):
        self.section = section
        self.strength = strength
        self.decay = decay

    def find_field(self, point: MPoint) -> float:
        closest = self._closest_point_on_segment(point)
        d = point.distance_to(closest)
        return self.strength / (1 + self.decay * d)

    def gradient(self, point: MPoint) -> MPoint:
        closest = self._closest_point_on_segment(point)
        direction = point.copy().subtract(closest)
        d = np.linalg.norm(direction.as_array())

        if d == 0:
            return MPoint(0, 0, 0)

        grad = direction.scale(self.strength * self.decay / ((1 + self.decay * d) ** 2)).normalise()
        return grad

    def _closest_point_on_segment(self, point: MPoint) -> MPoint:
        """Find the closest point on the section line segment."""
        a = self.section.start.as_array()
        b = self.section.end.as_array()
        p = point.as_array()

        ab = b - a
        ap = p - a
        denom = np.dot(ab, ab)
        t = np.clip(np.dot(ap, ab) / denom, 0, 1) if denom != 0 else 0.0
        closest = a + t * ab
        return MPoint.from_array(closest)

    def get_id(self):
        return id(self.section)  # Unique ID to support exclusion
