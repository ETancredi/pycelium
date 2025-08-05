# tropisms/field_finder.py

# Imports
from abc import ABC, abstractmethod # Abstract base class support
from core.point import MPoint # 3D point/vector class with operations
import numpy as np # Numerical operations

class FieldFinder(ABC):
    """Base class for any object that contributes a scalar field."""

    @abstractmethod
    def find_field(self, point: MPoint) -> float:
        """
        Computes and returns field strength at a given point.
        Args:
            point: MPoint where field is evaluated.
        Returns:
            float: field magnitude at that location.
        """
        pass # Must be implemented by subclasses

    def gradient(self, point: MPoint) -> MPoint:
        """
        Compute the gradient (vector of field change) at a given point.
        By default, returns zero vector (no gradient).
        Args:
            point: MPoint where gradient is evaluated.
        Returns:
            MPoint: vector indicating direction of greatest increase.
        """
        return MPoint(0, 0, 0)

    def get_id(self) -> int:
        """
        Return an identified for this field source.
        Used to exclude this source from self-interaction.
        Returns:
            int: unique ID, default -1 when not provided.
        """
        return -1


class PointFieldFinder(FieldFinder):
    """A simple field emitter from a fixed point (e.g., substrate or tip)."""

    def __init__(self, source: MPoint, strength: float = 1.0, decay: float = 1.0):
        """
        Initialise a point-based field source.
        Args:
            source: MPoint location of the emitter.
            strength: scalar strength coefficient
            decay: rate at which field decays with distance.
        """
        # Store a copy of the source point so original isn't mutated
        self.source = source.copy()
        # Scalar coefficient for field magnitude
        self.strength = strength
        # Decay factor: Field ~ strength / (1 + d * decay)
        self.decay = decay  

    def find_field(self, point: MPoint) -> float:
        """
        Compute field strength at 'point' based on inverse decay law.
        Args:
            point: MPoint where field is evaluated.
        Returns:
            float: computed field magnitude.
        """
        # Compute Euclidian distance from source to query point
        d = self.source.distance_to(point)
        # Return strength divided by (1 + decay * distance)
        return self.strength / (1.0 + self.decay * d)

    def gradient(self, point: MPoint) -> MPoint:
        """
        Approximate the gradient vector of the field at 'point'.
        Points in direction from source --> point, scaled by field slope.
        Args:
            point: MPoint where gradient is evaluated.
        Returns:
            MPoint: unit vector in gradient direction.
        """
        # Compute direction vector from source to point
        direction = point.copy().subtract(self.source)
        # Compute magnitude of that vector
        d = np.linalg.norm(direction.as_array())
        # If at the source exactly, gradient is zero
        if d == 0:
            return MPoint(0, 0, 0)
        return direction.scale(self.strength * self.decay / ((1 + self.decay * d) ** 2)).normalise()
        # Compute scalar factor: derivate of strength/(1+decay*d) wrt d
        # Scale the direction vector by this factor
        # Normalise to unit length to represent direction of steepest ascent
