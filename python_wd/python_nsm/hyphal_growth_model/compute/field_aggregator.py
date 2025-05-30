# compute/field_aggregator.py

from tropisms.field_finder import FieldFinder
from tropisms.sect_field_finder import SectFieldFinder
from core.section import Section
from core.point import MPoint
from core.options import ToggleableFloat
from typing import List
import numpy as np

class FieldAggregator:
    """
    Collects multiple FieldFinders into one composite source.
    Handles substrate and section influence.
    """

    def __init__(self):
        self.sources: List[FieldFinder] = []
        self.options = None

    def set_options(self, options):
        self.options = options

    def add_finder(self, finder: FieldFinder):
        self.sources.append(finder)

    def add_sections(self, sections: List[Section], strength=1.0, decay=1.0):
        for sec in sections:
            self.sources.append(SectFieldFinder(sec, strength=strength, decay=decay))

    def compute_field(self, point: MPoint, exclude_ids: List[int] = []) -> tuple[float, MPoint]:
        total_field = 0.0
        total_grad = MPoint(0, 0, 0)

        for source in self.sources:
            if source.get_id() in exclude_ids:
                continue

            # — radius‐based filtering (support ToggleableFloat) —
            if self.options:
                nr = self.options.neighbour_radius
                if isinstance(nr, ToggleableFloat):
                    if nr.enabled and nr.value > 0:
                        dist = point.distance_to(source.section.end)
                        if dist > nr.value:
                            continue
                else:
                    # legacy float/int
                    if nr > 0:
                        dist = point.distance_to(source.section.end)
                        if dist > nr:
                            continue

            total_field += source.find_field(point)
            grad = source.gradient(point)
            total_grad.add(grad)

        return total_field, total_grad.normalise()

    def compute_field_curvature(self, point: MPoint, epsilon=1.0) -> float:
        """
        Approximate curvature (Laplacian) of the scalar field at a point
        using finite differences. Returns a scalar.
        """
        base_value, _ = self.compute_field(point)

        offsets = [
            MPoint(epsilon, 0, 0), MPoint(-epsilon, 0, 0),
            MPoint(0, epsilon, 0), MPoint(0, -epsilon, 0),
            MPoint(0, 0, epsilon), MPoint(0, 0, -epsilon),
        ]

        laplace_sum = 0.0
        for offset in offsets:
            neighbor = point.copy().add(offset)
            neighbor_value, _ = self.compute_field(neighbor)
            laplace_sum += neighbor_value - base_value

        curvature = laplace_sum / (epsilon ** 2)
        return curvature
