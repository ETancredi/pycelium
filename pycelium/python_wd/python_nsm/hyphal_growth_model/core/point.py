# core/point.py

# Imports
import numpy as np # Array storage and vector maths

class MPoint:
    """3D point class with vector operations used in mycelium growth simulation."""
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
        Initialise with x, y, z as floats.
        Args:
            x, y, z: Coordinates in 3D space.
        """
        # Store coordinates in a NumPy array of type float64
        self.coords = np.array([x, y, z], dtype=np.float64)

    def copy(self):
        """
        Return a new independent MPoint with the same coordinates.
        This ensures modifications to the copy do not afect the original.
        """
        # Unpack coords into a new MPoint
        return MPoint(*self.coords)

    def distance_to(self, other) -> float:
        """
        Euclidean distance to another MPoint.
        Args:
            other: Another MPoint instance.
        Returns:
            float: √[(x2−x1)² + (y2−y1)² + (z2−z1)²]
        """
        # Compute vector difference and its L2 norm
        return np.linalg.norm(self.coords - other.coords)

    def normalise(self):
        """
        Convert this point's vector to unit length in-place.
        If the vector is zero-length, it remains unchanged.
        Returns:
            self: The same MPoint, now normalised.
        """
        norm = np.linalg.norm(self.coords) # Compute current vector magnitude
        if norm == 0:
            # Avoid division by zero: leave coords as-is
            return self
        # Divide each component by the norm
        self.coords /= norm
        return self

    def add(self, other):
        """
        In-place vector addition.
        Args:
            other: MPoint to add.
        Returns:
            self: Updated MPoint after addition.
        """
        # Component-wise addition of coordinate arrays
        self.coords += other.coords
        return self

    def subtract(self, other):
        """
        In place vector subtraction.
        Args:
            other: MPoint to subtract.
        Returns:
            self: Updated MPoint after subtraction.
        """
        # Component-wise subtraction
        self.coords -= other.coords
        return self

    def scale(self, factor):
        """
        In-place scalar multiplication.
        Args:
            factor: Number to multiply each component by.
        Returns:
            self: Scaled MPoint
        """
        self.coords *= factor
        return self

    def dot(self, other):
        """
        Dot product with another MPoint.
        Args: 
            other: MPoint for dot product.
        Returns:
            float: sum(self.coords[i] * other.coords[i] for i in 0..2)
        """
        # Use NumPy's dot to compute scalar product
        return float(np.dot(self.coords, other.coords))

    def cross(self, other):
        """Cross product with another MPoint (returns a new MPoint).
        Args:
            other: MPoint for cross product.
        Returns:
            MPoint: Vector perpendicular to self and other.
        """
        # Compute 3D cross product and wrap in MPoint
        return MPoint(*np.cross(self.coords, other.coords))

    def rotated_around(self, axis, angle_degrees):
        """
        Rotate this vector around a given axis by angle (degrees) in-place.
        Uses Rodrigues' rotation formula.
        Args:
            axis: MPoint representing rotation axis vector
            angle_degrees: Rotation angle in degrees
        Returns:
            self: Rotated MPoint.
        """
        # Convert degrees to radians
        angle_rad = np.deg2rad(angle_degrees)
        # Normalise axis vector
        axis_vec = axis.coords / np.linalg.norm(axis.coords)
        # Apply Rodrigues' formula: v_rot = v cosθ + (k×v) sinθ + k (k·v)(1−cosθ)
        self.coords = (
            self.coords * np.cos(angle_rad)
            + np.cross(axis_vec, self.coords) * np.sin(angle_rad)
            + axis_vec * (np.dot(axis_vec, self.coords)) * (1 - np.cos(angle_rad))
        )
        return self

    def __str__(self):
        """
        String representation of the point.
        Formats each coordinate to three decimal places.
        """
        return f"MPoint({self.coords[0]:.3f}, {self.coords[1]:.3f}, {self.coords[2]:.3f})"

    def to_list(self):
        """
        Convert coordinates to a standard Python list [x, y, z].
        Useful for JSON serialisation or simple inspection.
        """
        return self.coords.tolist()

    def as_array(self):
        """
        Return as NumPy array (for low-level operations).
        Caution: modifying the array will change this MPoint.
        """
        return self.coords

    @staticmethod
    def from_array(arr):
        """
        Construct an MPoint from a sequence (list/array) of at least 3 elements.
        Args:
            arr: Sequence with numeric values.
        Returns:
            MPoint: New point using arr[0], arr[1], arr[2].
        """
        # Unpack the first 3 elements into a new MPoint.
        return MPoint(*arr[:3])
