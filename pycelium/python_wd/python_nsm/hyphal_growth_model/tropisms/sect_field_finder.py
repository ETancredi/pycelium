# tropisms/sect_field_finder.py

# Imports
from tropisms.field_finder import FieldFinder # Abstract class for field sources
from core.point import MPoint # 3D point/vector class for ops like distance, add, subtract
from core.section import Section # Section class represents hyphal segment
import numpy as np # Numerical ops

class SectFieldFinder(FieldFinder):
    """
    Emits a scalar field based on distance from a line segment defined by a Section.
    Can be used for autotropism or self-avoidance.
    """
    def __init__(self, section: Section, strength: float = 1.0, decay: float = 1.0):
        """
        Initialise a SectFieldFinder.
        Args:
            section (Section): Segment whose line defines field source.
            strength (float): Scalar multiplier for field magnitude.
            decay (float): Rate at which field decays w/ distance.
        """
        self.section = section # Store target Section object
        self.strength = strength # Field strength coefficient
        self.decay = decay # Decay rate per unit distance

    def find_field(self, point: MPoint) -> float:
        """
        Compute scalar field strength at a given point.
            1. Find closest point on section's line segment to 'point'.
            2. Measure Euclidian distance 'd' from 'point' to closest point
            3. Return strength / (1 + decay * d)
        Args:
            point (MPoint): Location where field is evaluated.
        Returns:
            float: Scalar field value at 'point'.
        """
        closest = self._closest_point_on_segment(point) # Determine nearest point on segemnt to query point
        d = point.distance_to(closest) # Compute distance from query point to nearest point
        return self.strength / (1 + self.decay * d) # Return decayed field value

    def gradient(self, point: MPoint) -> MPoint:
        """
        Compute gradient vector of field at 'point'.
            1. Find closest point on segment.
            2. Compute direction vector from that point to 'point'.
            3. Calculate gradient magnitude = strength * decay / (1 + decay * d)^2.
            4. Scale direction vector by that magnitude and normalise.
        Args:
            point (MPoint): Location where gradient is evaluated.
        Returns:
            MPoint: Unit vector pointing in the direction of greatest increase.
        """
        closest = self._closest_point_on_segment(point) # Find projection of 'point' onto segment
        direction = point.copy().subtract(closest) # Build a vector from segment to query point
        d = np.linalg.norm(direction.as_array()) # Compute length

        if d == 0: # If exactly on the segment, gradient is undefined --> return zero vector
            return MPoint(0, 0, 0)

        # Compute gradient magnitude (derivatuve of field wrt distance)
        # Scale direction vector by gradient magnitude
        grad = direction.scale(self.strength * self.decay / ((1 + self.decay * d) ** 2)).normalise() # Normalise to unit length
        return grad 

    def _closest_point_on_segment(self, point: MPoint) -> MPoint:
        """
        Find the closest point on the section's line segment to a given point.
        Uses projection of the point onto the line AB, then clamps param t to [0,1].
        Args:
            point (MPoint): location to project.
        Returns:
            MPoint: closest point on segment AB.
        """
        # Convert Section start/end and query point to NumPy arrays
        a = self.section.start.as_array()
        b = self.section.end.as_array()
        p = point.as_array()
    
        # Vector from A to B, and from A to P
        ab = b - a
        ap = p - a
        # Denominator for projection forumila = |AB|^2
        denom = np.dot(ab, ab)
        # Compute raw projection factor t = (AP ⋅ AB) / (AB ⋅ AB)
        # Clip to [0,1] to stay w/in segment; handle degenerate AB length = 0
        t = np.clip(np.dot(ap, ab) / denom, 0, 1) if denom != 0 else 0.0
        closest = a + t * ab # Compute coords of closest point
        return MPoint.from_array(closest) # Return as a new MPoint

    def get_id(self):
        """
        Return a quniue identifier for this field source.
        Uses the Section object's ID to allow exclusion in aggregation
        Returns:
            int: Unique ID based on underlying Section.
        """
        return id(self.section)
