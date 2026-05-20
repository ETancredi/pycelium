# compute/drug_field.py

"""
Diffusing antifungal concentration field for Pycelium.

This module deliberately keeps the antifungal field separate from the existing
crowding / density field:

    - the density field asks "is this hyphal tip too crowded?"
    - the drug field asks "how inhibited is this hyphal tip by local antifungal?"

At each Pycelium growth step, main.step_simulation() calls diffuse_once() before
Mycel.step(). Then each active tip samples the already-diffused concentration at
its end point and converts that local concentration into a growth multiplier.

For now this is a 2D surface field because the current Pycelium runs and the
megaplate experiments are effectively plate-surface problems. A 3D agar-depth
version can be added later without changing the interface used by Mycel.step().
"""

# Imports
from __future__ import annotations  # Allow modern type hints while preserving compatibility

import logging  # Standard logging; quiet unless PYCELIUM_LOG_LEVEL enables it
from dataclasses import dataclass, field  # Lightweight containers for field configuration
from typing import Literal, Sequence  # Restricted string values and generic sequences

import matplotlib.pyplot as plt  # Used only for optional PNG export of the final field
import numpy as np  # NumPy arrays and vectorised finite-difference diffusion

from core.point import MPoint  # Pycelium point/vector class used for sampling tip positions

logger = logging.getLogger("pycelium.compute.drug_field")

# Boundary modes supported by this field.
#   noflux    -> approximate zero normal gradient, dC/dn = 0
#   absorbing -> edge concentration fixed to zero
#   constant  -> edge concentration fixed to a user-provided value
BoundaryMode = Literal["noflux", "absorbing", "constant"]


@dataclass
class DrugBoundary:
    """
    Boundary condition for one edge of the antifungal grid.

    Attributes:
        mode:
            Boundary-condition type. Use "noflux" for solid plate walls,
            "absorbing" for a sink-like edge, or "constant" for a fixed reservoir.
        value:
            Fixed concentration used only when mode == "constant".
    """

    mode: BoundaryMode = "noflux"  # Default to solid plate edge / no mass leaving domain
    value: float = 0.0             # Fixed edge concentration for constant boundaries


@dataclass
class DrugFieldConfig:
    """
    Numerical and biological settings for the antifungal field.

    The finite-difference update uses an explicit 2D diffusion scheme. For this
    scheme to be numerically stable, alpha = D * dt / dx^2 must be <= 0.25 for
    each numerical diffusion substep. The DrugField2D constructor checks this and
    raises a clear error if the requested values are unstable.
    """

    # Physical coordinate bounds. These are in the same coordinate units as MPoint.
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0

    # Grid spacing. Smaller dx gives a smoother field but increases memory/runtime.
    dx: float = 1.0

    # Diffusion parameters. diffusion_dt is normally opts.time_step.
    diffusion_coefficient: float = 0.05
    diffusion_dt: float = 1.0
    diffusion_substeps: int = 1

    # Optional first-order decay. Leave at 0.0 unless modelling drug degradation.
    decay_rate: float = 0.0

    # Baseline concentration used to initialise the whole field.
    initial_background_concentration: float = 0.0

    # Boundary conditions for each edge of the 2D plate field.
    left_boundary: DrugBoundary = field(default_factory=lambda: DrugBoundary("noflux", 0.0))
    right_boundary: DrugBoundary = field(default_factory=lambda: DrugBoundary("noflux", 0.0))
    bottom_boundary: DrugBoundary = field(default_factory=lambda: DrugBoundary("noflux", 0.0))
    top_boundary: DrugBoundary = field(default_factory=lambda: DrugBoundary("noflux", 0.0))


class DrugField2D:
    """
    Explicit finite-difference 2D antifungal diffusion field.

    Internally, the concentration array is indexed as:

        concentration[y_index, x_index]

    This follows normal NumPy image/grid convention, while Pycelium tip positions
    are sampled using their continuous x/y coordinates.
    """

    def __init__(self, config: DrugFieldConfig):
        """
        Create a new antifungal field from a DrugFieldConfig object.
        """
        # Store configuration for later use by diffusion, sampling, and exports.
        self.config = config

        # Validate basic geometry before allocating arrays.
        if config.dx <= 0.0:
            raise ValueError("drug_field_dx must be > 0.")
        if config.x_max <= config.x_min:
            raise ValueError("drug_field_x_max must be greater than drug_field_x_min.")
        if config.y_max <= config.y_min:
            raise ValueError("drug_field_y_max must be greater than drug_field_y_min.")
        if config.diffusion_substeps < 1:
            raise ValueError("drug_diffusion_substeps must be >= 1.")

        # Build coordinate axes. The +0.5*dx makes the upper bound inclusive despite
        # floating-point rounding, so x_max/y_max are represented on the grid.
        self.x_values = np.arange(config.x_min, config.x_max + 0.5 * config.dx, config.dx)
        self.y_values = np.arange(config.y_min, config.y_max + 0.5 * config.dx, config.dx)

        # Cache grid dimensions for boundary checks and sampling.
        self.nx = len(self.x_values)
        self.ny = len(self.y_values)

        # The five-point diffusion stencil and boundary copying need at least
        # one interior cell plus one boundary cell on each side.
        if self.nx < 3 or self.ny < 3:
            raise ValueError("Drug field must contain at least 3 grid cells in x and y.")

        # Allocate the concentration grid. Values are floats so diffusion remains smooth.
        self.concentration = np.full(
            shape=(self.ny, self.nx),
            fill_value=float(config.initial_background_concentration),
            dtype=np.float64,
        )

        # Confirm explicit scheme stability before the first diffusion step.
        self._validate_stability()

        # Apply edge conditions immediately so initial state is physically consistent.
        self.apply_boundaries()

    def _validate_stability(self) -> None:
        """
        Check the explicit diffusion stability criterion.

        For a 2D five-point stencil, a safe criterion is:

            alpha = D * dt / dx^2 <= 0.25

        Here dt means the numerical substep size, not necessarily the whole
        Pycelium growth time_step, because diffusion can be split into substeps.
        """
        # Convert one Pycelium diffusion step into the smaller numerical substep.
        numerical_dt = self.config.diffusion_dt / self.config.diffusion_substeps

        # Dimensionless diffusion number controlling explicit-scheme stability.
        alpha = self.config.diffusion_coefficient * numerical_dt / (self.config.dx ** 2)

        # Store alpha for debugging, logging, and potential later diagnostics.
        self.alpha = alpha

        # Refuse unstable settings rather than silently producing nonsense.
        if alpha > 0.25:
            raise ValueError(
                "Unstable drug diffusion settings: "
                f"alpha = D * dt / dx^2 = {alpha:.4f}, but alpha must be <= 0.25. "
                "Increase drug_diffusion_substeps, decrease drug_diffusion_coefficient, "
                "decrease drug_diffusion_dt, or increase drug_field_dx."
            )

    @classmethod
    def from_options(cls, opts) -> "DrugField2D":
        """
        Build a DrugField2D directly from the main Options dataclass.

        Keeping this conversion here prevents main.py from becoming cluttered
        with low-level field-construction details.
        """
        # Use one boundary mode for all four plate edges in the first implementation.
        # More detailed mixed boundaries can be added later by adding separate option keys.
        boundary = DrugBoundary(
            mode=getattr(opts, "drug_boundary_mode", "noflux"),
            value=getattr(opts, "drug_boundary_value", 0.0),
        )

        # Convert Options values into the field-specific config object.
        config = DrugFieldConfig(
            x_min=getattr(opts, "drug_field_x_min", getattr(opts, "x_min", -50.0)),
            x_max=getattr(opts, "drug_field_x_max", getattr(opts, "x_max", 50.0)),
            y_min=getattr(opts, "drug_field_y_min", getattr(opts, "y_min", -50.0)),
            y_max=getattr(opts, "drug_field_y_max", getattr(opts, "y_max", 50.0)),
            dx=getattr(opts, "drug_field_dx", 1.0),
            diffusion_coefficient=getattr(opts, "drug_diffusion_coefficient", 0.05),
            diffusion_dt=getattr(opts, "drug_diffusion_dt", getattr(opts, "time_step", 1.0)),
            diffusion_substeps=getattr(opts, "drug_diffusion_substeps", 1),
            decay_rate=getattr(opts, "drug_decay_rate", 0.0),
            initial_background_concentration=getattr(opts, "drug_initial_background_concentration", 0.0),
            left_boundary=boundary,
            right_boundary=boundary,
            bottom_boundary=boundary,
            top_boundary=boundary,
        )

        # Create and return the actual field object.
        return cls(config)

    def apply_boundaries(self) -> None:
        """
        Apply all four edge boundary conditions in-place.

        Corners are touched twice, but that is acceptable for this first simple
        boundary implementation. If mixed boundaries become biologically important
        later, we can define explicit corner precedence rules.
        """
        # Vertical edges are left/right columns.
        self._apply_vertical_boundary(edge="left", boundary=self.config.left_boundary)
        self._apply_vertical_boundary(edge="right", boundary=self.config.right_boundary)

        # Horizontal edges are bottom/top rows.
        self._apply_horizontal_boundary(edge="bottom", boundary=self.config.bottom_boundary)
        self._apply_horizontal_boundary(edge="top", boundary=self.config.top_boundary)

        # Guard against tiny negative values from roundoff or absorbing-edge interactions.
        np.maximum(self.concentration, 0.0, out=self.concentration)

    def _apply_vertical_boundary(self, edge: Literal["left", "right"], boundary: DrugBoundary) -> None:
        """
        Apply a boundary condition to the left or right edge of the grid.
        """
        # Choose the outer boundary column and the adjacent interior column.
        if edge == "left":
            boundary_col = 0
            neighbour_col = 1
        else:
            boundary_col = -1
            neighbour_col = -2

        # No-flux approximates dC/dx = 0 by copying the adjacent interior value.
        if boundary.mode == "noflux":
            self.concentration[:, boundary_col] = self.concentration[:, neighbour_col]

        # Absorbing boundary removes drug reaching the edge by fixing C = 0.
        elif boundary.mode == "absorbing":
            self.concentration[:, boundary_col] = 0.0

        # Constant boundary acts as a fixed concentration reservoir.
        elif boundary.mode == "constant":
            self.concentration[:, boundary_col] = boundary.value

        else:
            raise ValueError(f"Unknown drug boundary mode: {boundary.mode}")

    def _apply_horizontal_boundary(self, edge: Literal["bottom", "top"], boundary: DrugBoundary) -> None:
        """
        Apply a boundary condition to the bottom or top edge of the grid.
        """
        # Choose the outer boundary row and the adjacent interior row.
        if edge == "bottom":
            boundary_row = 0
            neighbour_row = 1
        else:
            boundary_row = -1
            neighbour_row = -2

        # No-flux approximates dC/dy = 0 by copying the adjacent interior value.
        if boundary.mode == "noflux":
            self.concentration[boundary_row, :] = self.concentration[neighbour_row, :]

        # Absorbing boundary fixes concentration to zero.
        elif boundary.mode == "absorbing":
            self.concentration[boundary_row, :] = 0.0

        # Constant boundary fixes concentration to the user-specified value.
        elif boundary.mode == "constant":
            self.concentration[boundary_row, :] = boundary.value

        else:
            raise ValueError(f"Unknown drug boundary mode: {boundary.mode}")

    def diffuse_once(self) -> None:
        """
        Advance the antifungal field by one Pycelium-level diffusion step.

        This method is called once before fungal growth in each simulation step.
        Internally, the method may use multiple smaller numerical substeps if
        drug_diffusion_substeps > 1.
        """
        # Use local names for readability and speed inside the substep loop.
        D = self.config.diffusion_coefficient
        dx = self.config.dx
        dt = self.config.diffusion_dt / self.config.diffusion_substeps
        alpha = D * dt / (dx ** 2)

        # Perform as many numerical substeps as requested.
        for _ in range(self.config.diffusion_substeps):
            # Enforce boundary conditions before computing the Laplacian.
            self.apply_boundaries()

            # Work from the old grid and write into a separate new grid to avoid
            # order-dependent updates.
            old = self.concentration
            new = old.copy()

            # Five-point Laplacian on the interior cells only.
            laplacian = (
                old[1:-1, 2:]   # right neighbour
                + old[1:-1, :-2] # left neighbour
                + old[2:, 1:-1]  # top neighbour
                + old[:-2, 1:-1] # bottom neighbour
                - 4.0 * old[1:-1, 1:-1]
            )

            # Explicit diffusion update: C(t+dt) = C(t) + alpha * Laplacian(C).
            new[1:-1, 1:-1] = old[1:-1, 1:-1] + alpha * laplacian

            # Optional first-order exponential decay of the drug.
            if self.config.decay_rate > 0.0:
                new[1:-1, 1:-1] *= np.exp(-self.config.decay_rate * dt)

            # Drug concentration cannot be negative.
            np.maximum(new, 0.0, out=new)

            # Swap in the updated grid and re-apply boundaries to finish the substep.
            self.concentration = new
            self.apply_boundaries()

    def set_rectangle(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        concentration: float,
    ) -> None:
        """
        Set a rectangular region to a fixed starting concentration.

        This is useful for manually initialising reservoirs, stripes, or plate sectors.
        The field is still free to diffuse after the initial condition is set.
        """
        # Identify grid columns/rows whose coordinate values fall inside the rectangle.
        x_mask = (self.x_values >= x_min) & (self.x_values <= x_max)
        y_mask = (self.y_values >= y_min) & (self.y_values <= y_max)

        # np.ix_ builds a rectangular 2D index from the row/column masks.
        self.concentration[np.ix_(y_mask, x_mask)] = concentration

        # Keep boundaries consistent after changing the grid.
        self.apply_boundaries()

    def set_vertical_sections(
        self,
        x_edges: Sequence[float],
        concentrations: Sequence[float],
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        """
        Initialise a megaplate-like set of vertical drug sections.

        Example:
            x_edges = [-50, -25, 0, 25, 50]
            concentrations = [0.0, 0.5, 1.0, 8.0]

        This creates four vertical zones:
            -50..-25 -> 0.0
            -25..0   -> 0.5
             0..25   -> 1.0
             25..50  -> 8.0
        """
        # There must be one more edge than there are section concentrations.
        if len(x_edges) != len(concentrations) + 1:
            raise ValueError("drug_initial_x_edges must have one more value than drug_initial_concentrations.")

        # Default to the full y extent of the field if no y-range was provided.
        if y_min is None:
            y_min = self.config.y_min
        if y_max is None:
            y_max = self.config.y_max

        # Fill each neighbouring pair of x edges with the matching concentration.
        for left, right, concentration in zip(x_edges[:-1], x_edges[1:], concentrations):
            self.set_rectangle(left, right, y_min, y_max, concentration)

    def sample(self, point_or_x: MPoint | float, y: float | None = None) -> float:
        """
        Bilinearly sample the field at a continuous Pycelium coordinate.

        Args:
            point_or_x:
                Either an MPoint with x/y coordinates or a raw x coordinate.
            y:
                Raw y coordinate when point_or_x is a float.

        Returns:
            Local antifungal concentration at that coordinate.
        """
        # Accept either sample(MPoint(...)) or sample(x, y) for convenience.
        if isinstance(point_or_x, MPoint):
            x_coord = float(point_or_x.coords[0])
            y_coord = float(point_or_x.coords[1])
        else:
            if y is None:
                raise ValueError("sample(x, y) requires both x and y coordinates.")
            x_coord = float(point_or_x)
            y_coord = float(y)

        # Clamp samples outside the drug field to the nearest edge. This prevents
        # crashes if a tip slightly exceeds the field bounds before volume pruning.
        x_coord = min(max(x_coord, self.config.x_min), self.config.x_max)
        y_coord = min(max(y_coord, self.config.y_min), self.config.y_max)

        # Convert physical coordinates into fractional grid indices.
        gx = (x_coord - self.config.x_min) / self.config.dx
        gy = (y_coord - self.config.y_min) / self.config.dx

        # Lower-left integer grid cell. Clamp in case non-divisible bounds/dx cause
        # a coordinate exactly at the upper bound to round onto the final index.
        x0 = min(max(int(np.floor(gx)), 0), self.nx - 1)
        y0 = min(max(int(np.floor(gy)), 0), self.ny - 1)

        # Upper/right neighbouring cell, clamped at grid edge.
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)

        # Fractional weights within the cell.
        wx = gx - x0
        wy = gy - y0

        # Fetch the four surrounding concentrations.
        c00 = self.concentration[y0, x0]
        c10 = self.concentration[y0, x1]
        c01 = self.concentration[y1, x0]
        c11 = self.concentration[y1, x1]

        # Linear interpolation in x along bottom and top grid rows.
        bottom = (1.0 - wx) * c00 + wx * c10
        top = (1.0 - wx) * c01 + wx * c11

        # Linear interpolation in y between the two row-interpolated values.
        return float((1.0 - wy) * bottom + wy * top)

    @staticmethod
    def growth_multiplier(
        concentration: float,
        mic: float,
        hill_coefficient: float = 4.0,
        min_multiplier: float = 0.0,
    ) -> float:
        """
        Convert local drug concentration into a growth-rate multiplier.

        The response is a Hill-style inhibition curve:

            multiplier = min + (1 - min) / (1 + (C / MIC)^h)

        Interpretation:
            C << MIC  -> multiplier approaches 1.0, so growth is nearly normal.
            C == MIC  -> multiplier is halfway between 1.0 and min_multiplier.
            C >> MIC  -> multiplier approaches min_multiplier.

        MIC is therefore the trait that future resistant mutants can change.
        """
        # A zero or negative MIC means the tip cannot tolerate any drug.
        if mic <= 0.0:
            return float(min_multiplier)

        # Ensure numerical/biological bounds are sensible.
        C = max(float(concentration), 0.0)
        h = max(float(hill_coefficient), 1e-9)
        floor = min(max(float(min_multiplier), 0.0), 1.0)

        # Compute the Hill inhibition response.
        response = 1.0 / (1.0 + (C / float(mic)) ** h)

        # Blend from full growth down to the configured lower bound.
        return float(floor + (1.0 - floor) * response)

    def export_npy(self, path: str) -> None:
        """
        Save the final concentration array as a NumPy binary file.
        """
        np.save(path, self.concentration)
        logger.info("Drug field NPY exported: %s", path)

    def export_csv(self, path: str) -> None:
        """
        Save the final concentration array as a CSV file.
        """
        np.savetxt(path, self.concentration, delimiter=",", fmt="%.6f")
        logger.info("Drug field CSV exported: %s", path)

    def export_png(self, path: str, cmap: str = "magma") -> None:
        """
        Save the final concentration array as a heatmap PNG.
        """
        plt.figure(figsize=(7, 5))
        plt.imshow(
            self.concentration,
            origin="lower",
            extent=[self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max],
            aspect="auto",
            cmap=cmap,
        )
        plt.colorbar(label="Antifungal concentration")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Final antifungal field")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        logger.info("Drug field PNG exported: %s", path)
