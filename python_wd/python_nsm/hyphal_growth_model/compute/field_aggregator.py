# compute/field_aggregator.py

# Imports
from tropisms.field_finder import FieldFinder # Base class for field generators
from tropisms.sect_field_finder import SectFieldFinder # Field generator based on a section
from core.section import Section # Mycelial segment
from core.point import MPoint # 3D point/vector representation
from typing import List # Type hinting for lists

# Class that aggregates multiple field sources
class FieldAggregator:
    """
    Collects multiple FieldFinders into one composite source.
    Handles substrate and section influence.
    """

    def __init__(self):
        self.sources: List[FieldFinder] = [] # List of all field sources to consider
        self.options = None # Placeholder for optional configuration 

    # Sets global field computation options (from simulation settings)
    def set_options(self, options):
        self.options = options

    # Add a generic fieldfinder object to the source list
    def add_finder(self, finder: FieldFinder):
        self.sources.append(finder)

    # Add a list of Section objects as field sources using SectFieldFinder wrappers
    def add_sections(self, sections: List[Section], strength=1.0, decay=1.0):
        for sec in sections:
            self.sources.append(SectFieldFinder(sec, strength=strength, decay=decay))

    # Computes the total field strength and gradient vector at a given point
    def compute_field(self, point: MPoint, exclude_ids: List[int] = []) -> tuple[float, MPoint]:
        total_field = 0.0 # Accumulate scala field values
        total_grad = MPoint(0, 0, 0) # Accumulate gradient vector

        for source in self.sources:
            if source.get_id() in exclude_ids: # Skip excluded sources
                continue

            # Apply neighbour-radius constaint if defined in options
            if self.options and self.options.neighbour_radius > 0:
                dist = point.distance_to(source.section.end) # Distance to section end
                if dist > self.options.neighbour_radius:
                    continue # Ignore if too far away

            total_field += source.find_field(point) # Add scalar field contribution
            grad = source.gradient(point) # Get gradient vector from this source
            total_grad.add(grad) # Accumulate gradients

        return total_field, total_grad.normalise() # Return scalar + unit gradient vector

    # Computes an approximate curvature (second spatial derivative) of the scalar field
    def compute_field_curvature(self, point: MPoint, epsilon=1.0) -> float:
        """
        Approximate curvature (Laplacian) of the scalar field at a point
        using finite differences. Returns a scalar.
        """
        base_value, _ = self.compute_field(point) # Field value at central point

        # Define symmetric offsets along X, Y, and Z axes
        offsets = [
            MPoint(epsilon, 0, 0), MPoint(-epsilon, 0, 0),
            MPoint(0, epsilon, 0), MPoint(0, -epsilon, 0),
            MPoint(0, 0, epsilon), MPoint(0, 0, -epsilon),
        ]

        laplace_sum = 0.0 # Sum of second differences

        # Approximate Laplacian: ∇²f ≈ sum of (f(x+ε) - f(x))
        for offset in offsets:
            neighbor = point.copy().add(offset) # Create nearby point
            neighbor_value, _ = self.compute_field(neighbor) # Get field value at offset
            laplace_sum += neighbor_value - base_value # Difference from center

        curvature = laplace_sum / (epsilon ** 2) # Scale by ε² to approximate curvature
        return curvature # Return scalar field curvature at the point
