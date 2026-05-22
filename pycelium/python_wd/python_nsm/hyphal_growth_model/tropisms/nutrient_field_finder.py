# tropisms/nutrient_field_finder.py

# Imports
from tropisms.field_finder import FieldFinder # Abstract base class for field sources
from core.point import MPoint # 3D point/vector class for geometry ops
import numpy as np # Numerical ops

class NutrientFieldFinder(FieldFinder):
    """
    FieldFinder subclass representing a nutrient attractor or repellent at a fixed location in space.
    """
    def __init__(self, location: MPoint, strength=1.0, decay=1.0, repulsive=False):
        """
        Initialise a NutrientFieldFinder.
        Args:
            location (MPoint): position of nutrient source.
            strength (float): Base magnitude of the field.
            decay (float): Decay rate of field strength w/ distance.
            repulsive (bool): if True, field is negative (repellent), otherwise positive (attractive).
        """
        self.location = location # Store fixed source location
        self.strength = strength # Scalar coefficient for field magnitude
        self.decay = decay # Rate of decay w/ distance
        self.repulsive = repulsive # Flag for whether this is an attractive or repulsive force

    def find_field(self, point: MPoint) -> float:
        """
        Compute scalar field strength at a given point.
        Field allows an inverse decay law: ±strength / (1 + decay * distance)
        Args:
            point (MPoint): Query location.
        Returns:
            float: Positive for attraction, negative for repulsion.
        """
        d = point.distance_to(self.location) # Compute Euclidian siatnce from the source to query point
        if self.repulsive: 
            return -self.strength / (1 + self.decay * d) # Negative field for repulsion
        return self.strength / (1 + self.decay * d) # Positive field for attraction

    def gradient(self, point: MPoint) -> MPoint:
        """
        Compute the unit gradient vector of the field at a point.
        Gradient magnitude is derivative of field wrt distance:
            d/dx [strength/(1+decay*d)] = ±(strength*decay)/((1+decay*d)^2)
        Args:
            point (MPoint): Location where gradient is evaluated.
        Returns:
            MPoint: Normalised direction of steepest ascent (or descent if repulsive).
        """
        direction = point.copy().subtract(self.location) # Vector pointing from source to query point
        d = np.linalg.norm(direction.as_array()) # Compute its magnitude
        if d == 0:
            return MPoint(0, 0, 0) # At the exact source, gradient is undefined-return zero vector
        grad_mag = self.strength * self.decay / ((1 + self.decay * d) ** 2) # Compute gradient magnitude (+)
        grad = direction.scale(-grad_mag if self.repulsive else grad_mag) # If repulsive, flip sign so gradient points downhill
        return grad.normalise() # Scale direction vector by magnitude and normalise

    def get_id(self):
        """ 
        Return a unique identifier for this source instance.
        Used by FieldAggregator to optionally exclude self-interaction.
        """
        return id(self)
