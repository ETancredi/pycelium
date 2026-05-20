# tropisms/substrate.py

# Imports
from core.point import MPoint # 3D point/vector ops
from tropisms.field_finder import FieldFinder # Base FieldFinder abstract class
import numpy as np # Numerical ops

class LinearSubstrate(FieldFinder):
    """
    Represents a linear substrate (e.g. a straight nutrient source).
    Emits a scalar field that decays with perpendicular distance from the line.
    """

    def __init__(self, origin: MPoint, direction: MPoint, strength: float = 1.0, decay: float = 1.0):
        """
        Initialise the linear substrate.
        Args:
            origin (MPoint): a point on the substrate line.
            direction (MPoint): direction vector of the line; will be normalised.
            strength (float): base field magnitude on the line.
            decay (float): rate of decay w/ perpendicular distance.
        """
        self.origin = origin.copy() # Store a copy of origin point so original isn't modified
        self.direction = direction.copy().normalise() # Normalise direction vector to unit length and store
        self.strength = strength # Store scalar field strenth param
        self.decay = decay # Store decay rate for distance-based attenuation

    def find_field(self, point: MPoint) -> float:
        """
        Compute scalar field strengt at query point.
        Field law: strength / (1 + decay * d_perp),
        where d_perp is the perpendicular distance to the substrate line.
        Args:
            point (MPoint): location where field is evaluated.
        Returns:
            float: Scalar field magnitude at 'point'.
        """
        delta = point.copy().subtract(self.origin) # Compute vector from origin to query point
        projection = self.direction.copy().scale(delta.dot(self.direction)) # Project delta onto substrate direction
        perpendicular = delta.subtract(projection) # Compute perpendicular component: delta minus its projection
        d = np.linalg.norm(perpendicular.as_array()) # Compute perpendicular distance magnitude
        return self.strength / (1 + self.decay * d) # Return decayed field strength based on perpendicular distance

    def gradient(self, point: MPoint) -> MPoint:
        """
        Compute field gradient vector at a query point.
        Gradient points toward or away from line (perp direction),
        with magnitude equal to deriv of scalar field.
        Args:
            point (MPoint): location where gradient is evaluated.
        Returns:
            MPoint: Unit vector in direction of greatest increase (or decrease)
        """
        delta = point.copy().subtract(self.origin) # Compute vector from origin to query point
        projection = self.direction.copy().scale(delta.dot(self.direction)) # Project delta onto substrate direction
        perpendicular = delta.subtract(projection) # Compute perpendicular component
        d = np.linalg.norm(perpendicular.as_array()) # Compute perpendicular distance

        if d == 0: # If ecactly on the line, gradient is zero vector
            return MPoint(0, 0, 0)

        # Compute gradient magnitude (derivative of strength / (1 + decay * d))
        # Scale perpendicular vector and normalise
        grad = perpendicular.scale(-self.strength * self.decay / ((1 + self.decay * d) ** 2)).normalise()
        return grad
