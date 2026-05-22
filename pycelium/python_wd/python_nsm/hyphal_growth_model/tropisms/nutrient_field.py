# tropsisms/nutrient_field.py

# Imports
from core.point import MPoint # 3D point/vector class with ops
import numpy as np # Exponential decay evaluation

class NutrientField:
    """
    Computes a vector field representing nutrient attraction or repulsion 
    from one or more point sources in 2D (x,y) plane, returning a 3D MPoint 
    with only X and Y components non-zero.
    """
    def __init__(self, attractors=None, decay=0.05):
        """
        Initialise the NutrientField.
        Args:
            attractors (list of ((x,y), strength) tuples:
                Positions and strengths of nutrient sources.
            decay (float): Exponential decay rate of nutrient influences with distance
        """
        # If no attractors list provided, default to empty list
        self.sources = attractors or []
        # Store the decay constant for influence fall-off
        self.decay = decay

    def compute(self, point: MPoint) -> MPoint:
        """
        Compute the net nutrient influence vector at a query point.
        Args:
            point (MPoint): The location where we evaluate the nutrient field.
        Returns:
            MPoint: A vector whose direction is sum of influences and whose 
                    magnitude is the sum of each source's decayed strength
        """
        # Start with zero vector
        total = MPoint(0, 0, 0)
        # Loop over each nutrient source
        for (x, y), strength in self.sources:
            # Build vector from the query point to the source
            # Note: use coords attribute if MPoint has no direct x,y properties
            vec = MPoint(x - point.x, y - point.y, 0)
            # Compute distance from source to point
            # MPoint does not define magnitude(); use Euclician norm directly
            dist = vec.magnitude()
            # Skip extremely close distances to avoid numerical instability
            if dist > 1e-3:
                # Compute decayed influence: strength * exp(-decay * distance)
                influence = strength * np.exp(-self.decay * dist)
                # Normalise direction vector and scale by influence 
                # vec.normalise() makes unit vector; .scale() multiplies by influence
                total.add(vec.normalise().scale(influence))
        # Return the accumulated influence vector
        return total
