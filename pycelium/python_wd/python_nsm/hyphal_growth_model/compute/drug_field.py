# compute/drug_field.py

"""
Diffusing antifungal concentration field for Pycelium.

This module deliberately keeps the antifungal field separate from the existing
crowding / density field:

    - the density field asks "is this hyphal tip too crowded?"
    - the drug field asks "how inhibited is this hyphal tip by local antifungal?"

At each Pycelium growth step, main.step_simulation() calls diffuse_once() before
Mycel.step(). Then each active tip samples the already-diffused concentration at
its end point and converts that local concentration into a local pharmacodynamic
growth rate.

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
from matplotlib.colors import LinearSegmentedColormap  # Custom white-to-dark-blue drug heatmap
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

        # Optional fixed source mask. This is used by square-perimeter fields where
        # the outer drug-loaded band should behave like a maintained reservoir.
        # If this remains None, all cells are free to diffuse normally after setup.
        self.fixed_source_mask = None
        self.fixed_source_value = None

        # Snapshot of the configured starting field. main.setup_simulation()
        # refreshes this after any user-selected initial condition is painted on.
        # It lets us export drug_field_initial.* for debugging, so final heatmaps
        # are not mistaken for the starting concentration pattern.
        self.initial_concentration = None

        # Biomass-hindered diffusion diagnostics.  These are refreshed during
        # diffuse_once(mycel=..., opts=...) when the biofilm barrier option is on,
        # and exported at the end of a run for visual/quantitative inspection.
        self.last_biomass_barrier = np.zeros_like(self.concentration, dtype=np.float64)
        self.last_effective_diffusion = np.full_like(
            self.concentration,
            fill_value=float(config.diffusion_coefficient),
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

    def enforce_fixed_sources(self) -> None:
        """
        Re-impose any maintained antifungal source cells.

        For an outside-in square field, the outer square frame can either be an
        initial bolus that is allowed to dilute, or a maintained source that is
        reset to the source concentration after every diffusion substep. This
        method implements the maintained-source behaviour when configured.
        """
        # If no maintained source has been configured, there is nothing to do.
        if self.fixed_source_mask is None:
            return

        # Refill all source cells to the configured concentration.
        self.concentration[self.fixed_source_mask] = self.fixed_source_value

    def build_hyphal_biomass_barrier(
        self,
        mycel,
        radius: float,
        length_scale: float,
        include_dead: bool = True,
    ) -> np.ndarray:
        """
        Rasterise the current hyphal network into a 2D biomass barrier grid.

        The resulting array is dimensionless.  Larger values indicate denser
        hyphal biomass and therefore stronger physical obstruction to antifungal
        diffusion.  Each hyphal subsegment contributes along its length and over
        a small radial neighbourhood, giving a smooth local barrier rather than
        a single-cell line.
        """
        barrier = np.zeros_like(self.concentration, dtype=np.float64)

        if mycel is None:
            return barrier

        radius = max(float(radius), self.config.dx)
        length_scale = max(float(length_scale), 1e-12)

        # Neighbourhood size in grid cells around each sampled hyphal point.
        cell_radius = max(1, int(np.ceil(radius / self.config.dx)))

        for section in mycel.get_all_segments():
            if getattr(section, "is_dead", False) and not include_dead:
                continue

            subsegments = getattr(section, "subsegments", [])
            for start, end in subsegments:
                p0 = np.asarray(start.coords[:2], dtype=float)
                p1 = np.asarray(end.coords[:2], dtype=float)
                delta = p1 - p0
                segment_length = float(np.linalg.norm(delta))

                # Ignore zero-length placeholders and fully stalled entries.
                if segment_length < 1e-12:
                    continue

                # Sample along the subsegment finely enough that no grid cells are
                # skipped. Each sample receives a length contribution so total
                # biomass scales with true hyphal path length.
                n_samples = max(2, int(np.ceil(segment_length / max(self.config.dx * 0.5, 1e-12))) + 1)
                sample_weight = segment_length / max(n_samples - 1, 1)

                for t in np.linspace(0.0, 1.0, n_samples):
                    x, y = p0 + t * delta
                    if (
                        x < self.config.x_min - radius
                        or x > self.config.x_max + radius
                        or y < self.config.y_min - radius
                        or y > self.config.y_max + radius
                    ):
                        continue

                    cx = int(round((x - self.config.x_min) / self.config.dx))
                    cy = int(round((y - self.config.y_min) / self.config.dx))

                    x0 = max(cx - cell_radius, 0)
                    x1 = min(cx + cell_radius + 1, self.nx)
                    y0 = max(cy - cell_radius, 0)
                    y1 = min(cy + cell_radius + 1, self.ny)

                    if x0 >= x1 or y0 >= y1:
                        continue

                    xs = self.x_values[x0:x1]
                    ys = self.y_values[y0:y1]
                    dx = xs[None, :] - x
                    dy = ys[:, None] - y
                    distance = np.sqrt(dx * dx + dy * dy)

                    # Compact linear kernel: strongest on the hypha, tapering to
                    # zero at the configured radius.
                    weights = np.maximum(0.0, 1.0 - distance / radius)
                    if np.any(weights):
                        barrier[y0:y1, x0:x1] += (sample_weight / length_scale) * weights

        return barrier

    def effective_diffusion_from_biomass(
        self,
        biomass_barrier: np.ndarray,
        strength: float,
        min_fraction: float,
    ) -> np.ndarray:
        """
        Convert dimensionless hyphal biomass into a local diffusion coefficient.

        D_eff = D * [f_min + (1 - f_min) * exp(-strength * biomass)]

        This gives normal diffusion where biomass is absent and progressively
        reduced diffusion in dense mycelium, bounded below by f_min * D.
        """
        base_D = float(self.config.diffusion_coefficient)
        strength = max(float(strength), 0.0)
        min_fraction = min(max(float(min_fraction), 0.0), 1.0)
        barrier = np.maximum(np.asarray(biomass_barrier, dtype=np.float64), 0.0)
        fraction = min_fraction + (1.0 - min_fraction) * np.exp(-strength * barrier)
        return base_D * fraction

    def _diffuse_variable_diffusion_substep(self, effective_diffusion: np.ndarray, dt: float) -> np.ndarray:
        """
        One explicit conservative diffusion substep with spatially varying D.

        This finite-volume-style update computes fluxes across cell faces using
        the mean D_eff of neighbouring cells.  When D_eff is uniform, it reduces
        to the original five-point diffusion stencil.
        """
        old = self.concentration
        new = old.copy()
        delta = np.zeros_like(old, dtype=np.float64)
        dx2 = self.config.dx ** 2

        # Fluxes across vertical faces between neighbouring x cells.
        D_x = 0.5 * (effective_diffusion[:, 1:] + effective_diffusion[:, :-1])
        flux_x = D_x * (old[:, 1:] - old[:, :-1]) / dx2
        delta[:, :-1] += flux_x
        delta[:, 1:] -= flux_x

        # Fluxes across horizontal faces between neighbouring y cells.
        D_y = 0.5 * (effective_diffusion[1:, :] + effective_diffusion[:-1, :])
        flux_y = D_y * (old[1:, :] - old[:-1, :]) / dx2
        delta[:-1, :] += flux_y
        delta[1:, :] -= flux_y

        # Update only the whole grid; boundary conditions are applied before and
        # after this method by diffuse_once().
        new = old + dt * delta

        if self.config.decay_rate > 0.0:
            new *= np.exp(-self.config.decay_rate * dt)

        np.maximum(new, 0.0, out=new)
        return new

    def diffuse_once(self, mycel=None, opts=None) -> None:
        """
        Advance the antifungal field by one Pycelium-level diffusion step.

        If opts.drug_diffusion_hindered_by_biomass is True and a Mycel object is
        supplied, dense hyphal biomass reduces the local effective diffusion
        coefficient before each diffusion update.  This models biofilm-like
        penetration limitation in dense aspergilloma/mycelial networks.
        """
        dx = self.config.dx
        dt = self.config.diffusion_dt / self.config.diffusion_substeps
        use_biomass_hindrance = bool(
            opts is not None
            and mycel is not None
            and getattr(opts, "drug_diffusion_hindered_by_biomass", False)
        )

        if use_biomass_hindrance:
            biomass_barrier = self.build_hyphal_biomass_barrier(
                mycel=mycel,
                radius=getattr(opts, "drug_hyphal_barrier_radius", 3.0),
                length_scale=getattr(opts, "drug_hyphal_barrier_length_scale", 20.0),
                include_dead=getattr(opts, "drug_hyphal_barrier_include_dead", True),
            )
            effective_diffusion = self.effective_diffusion_from_biomass(
                biomass_barrier=biomass_barrier,
                strength=getattr(opts, "drug_hyphal_barrier_strength", 2.0),
                min_fraction=getattr(opts, "drug_hyphal_barrier_min_diffusion_fraction", 0.05),
            )
            self.last_biomass_barrier = biomass_barrier
            self.last_effective_diffusion = effective_diffusion
        else:
            biomass_barrier = np.zeros_like(self.concentration, dtype=np.float64)
            effective_diffusion = np.full_like(
                self.concentration,
                fill_value=float(self.config.diffusion_coefficient),
                dtype=np.float64,
            )
            self.last_biomass_barrier = biomass_barrier
            self.last_effective_diffusion = effective_diffusion

        # Perform as many numerical substeps as requested.
        for _ in range(self.config.diffusion_substeps):
            self.apply_boundaries()
            self.enforce_fixed_sources()

            if use_biomass_hindrance:
                new = self._diffuse_variable_diffusion_substep(effective_diffusion, dt)
            else:
                D = self.config.diffusion_coefficient
                alpha = D * dt / (dx ** 2)
                old = self.concentration
                new = old.copy()
                laplacian = (
                    old[1:-1, 2:]
                    + old[1:-1, :-2]
                    + old[2:, 1:-1]
                    + old[:-2, 1:-1]
                    - 4.0 * old[1:-1, 1:-1]
                )
                new[1:-1, 1:-1] = old[1:-1, 1:-1] + alpha * laplacian
                if self.config.decay_rate > 0.0:
                    new[1:-1, 1:-1] *= np.exp(-self.config.decay_rate * dt)
                np.maximum(new, 0.0, out=new)

            self.concentration = new
            self.apply_boundaries()
            self.enforce_fixed_sources()

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

    def set_square_perimeter_source(
        self,
        border_width: float,
        source_concentration: float,
        interior_concentration: float | None = None,
        maintain_source: bool = True,
    ) -> None:
        """
        Initialise an outside-in square antifungal field.

        The outer square frame is set to source_concentration and the centre can
        optionally be reset to interior_concentration first. Diffusion then moves
        drug inward from all four sides, rather than from right to left as in the
        megaplate-style vertical-section setup.

        Args:
            border_width:
                Physical width of the drug-loaded outer frame, in the same
                coordinate units as the Pycelium simulation.
            source_concentration:
                Antifungal concentration assigned to the outer square frame.
            interior_concentration:
                Optional concentration assigned to the entire field before the
                perimeter is painted on. Use 0.0 for a drug-free centre.
            maintain_source:
                If True, the outer frame is reset to source_concentration after
                every diffusion substep, mimicking a replenished source/reservoir.
                If False, it is only an initial condition and will dilute over time.
        """
        # Validate the requested square-frame width. A non-positive width would
        # produce no source region and make the option misleading.
        if border_width <= 0.0:
            raise ValueError("drug_square_perimeter_width must be > 0.")

        # Optionally reset the full field first. For the outside-in test config,
        # this makes the centre initially drug-free.
        if interior_concentration is not None:
            self.concentration[:, :] = float(interior_concentration)

        # Work out distance from each grid coordinate to its nearest field edge.
        x_dist_to_edge = np.minimum(
            self.x_values - self.config.x_min,
            self.config.x_max - self.x_values,
        )
        y_dist_to_edge = np.minimum(
            self.y_values - self.config.y_min,
            self.config.y_max - self.y_values,
        )

        # A cell belongs to the square source frame if it is within border_width
        # of any of the four outer edges. np.ix_ is not needed here because the
        # row/column masks are broadcast to a full 2D mask.
        source_mask = (
            (x_dist_to_edge[None, :] <= border_width)
            | (y_dist_to_edge[:, None] <= border_width)
        )

        # Paint the outer square frame onto the field.
        self.concentration[source_mask] = float(source_concentration)

        # Either store the mask as a maintained source, or clear any previous
        # maintained source so this acts as a one-off initial condition.
        if maintain_source:
            self.fixed_source_mask = source_mask
            self.fixed_source_value = float(source_concentration)
        else:
            self.fixed_source_mask = None
            self.fixed_source_value = None

        # Keep the source and boundaries consistent immediately after setup.
        # Source cells are applied last so the outside frame remains loaded.
        self.apply_boundaries()
        self.enforce_fixed_sources()

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
    def pharmacodynamic_growth_rate(
        concentration: float,
        mic: float,
        max_growth_rate: float,
        min_growth_rate: float,
        hill_coefficient: float = 4.0,
    ) -> float:
        """
        Convert local antifungal concentration into an absolute growth rate.

        This implements the MIC-centred pharmacodynamic relationship:

            Ψ = Ψmax - ((Ψmax - Ψmin) * (A / MIC)^k)
                       / ((A / MIC)^k - (Ψmin / Ψmax))

        where:
            Ψmax = maximum growth rate in the absence of drug.
            Ψmin = minimum/asymptotic growth rate at very high drug.
            A    = local antifungal concentration.
            MIC  = concentration at which growth crosses zero.
            k    = Hill coefficient controlling response steepness.

        The important calibration property is:

            A = MIC  ->  Ψ = 0

        provided Ψmax > 0 and Ψmin < 0. Therefore, unlike the older multiplier
        curve, MIC now really means the concentration at which net growth stops.
        Values above MIC produce negative raw growth rates; Mycel.step() decides
        whether those non-positive rates should stall the tip or kill it.
        """
        # Defensive conversion makes JSON-loaded ints/floats behave consistently.
        A = max(float(concentration), 0.0)       # Antifungal concentration cannot be negative
        MIC = float(mic)                         # Tip-specific MIC / tolerance trait
        psi_max = float(max_growth_rate)         # Ψmax: drug-free growth rate
        psi_min = float(min_growth_rate)         # Ψmin: high-drug asymptote, normally negative
        k = max(float(hill_coefficient), 1e-9)   # Avoid divide-by-zero-like k values

        # A positive Ψmax is required because the equation contains Ψmin / Ψmax.
        if psi_max <= 0.0:
            raise ValueError("drug pharmacodynamic model requires growth_rate / Ψmax > 0.")

        # MIC must be positive for A / MIC to be meaningful.
        if MIC <= 0.0:
            raise ValueError("drug_wildtype_mic and section.drug_mic must be > 0.")

        # Ψmin must be negative if MIC is to be the zero-growth concentration.
        # If Ψmin were zero, the formula is singular at A=0 and cannot encode MIC.
        if psi_min >= 0.0:
            raise ValueError(
                "drug_min_growth_rate / Ψmin must be < 0 for the pharmacodynamic "
                "MIC model. Example: with growth_rate=1.0, use drug_min_growth_rate=-1.0."
            )

        # Dimensionless drug exposure relative to MIC.
        relative_exposure = (A / MIC) ** k

        # Denominator uses Ψmin / Ψmax. This is the form that guarantees Ψ=0
        # when A == MIC. Reversing the fraction would not satisfy the MIC condition.
        denominator = relative_exposure - (psi_min / psi_max)

        # The denominator should be positive under the validity checks above, but
        # guard against pathological floating-point states anyway.
        if denominator <= 0.0:
            return psi_min

        # Return the signed raw growth rate. Downstream code can clamp or kill.
        return float(
            psi_max
            - ((psi_max - psi_min) * relative_exposure) / denominator
        )

    @staticmethod
    def growth_multiplier(
        concentration: float,
        mic: float,
        hill_coefficient: float = 4.0,
        min_multiplier: float = 0.0,
    ) -> float:
        """
        Convert local drug concentration into a legacy growth-rate multiplier.

        This is retained for backward compatibility with early drug-field tests.
        For MIC-calibrated runs, prefer pharmacodynamic_growth_rate(), because
        that model makes C == MIC correspond to zero growth.

        The legacy response is a Hill-style inhibition curve:

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

    def capture_initial_concentration(self) -> None:
        """
        Store a copy of the fully configured starting field.

        This should be called after the selected initial condition has been
        applied, but before the first diffusion/growth step. Exporting this
        snapshot makes it obvious whether a run started as uniform drug,
        vertical sections, or an outside-in square source.
        """
        self.initial_concentration = self.concentration.copy()

    def diagnostic_summary(self, array=None) -> dict:
        """
        Return compact field diagnostics for console/log sanity checks.
        """
        C = self.concentration if array is None else array
        centre_y = C.shape[0] // 2
        centre_x = C.shape[1] // 2
        return {
            "shape": C.shape,
            "min": float(np.min(C)),
            "max": float(np.max(C)),
            "mean": float(np.mean(C)),
            "centre": float(C[centre_y, centre_x]),
            "bottom_left": float(C[0, 0]),
            "bottom_right": float(C[0, -1]),
            "top_left": float(C[-1, 0]),
            "top_right": float(C[-1, -1]),
        }

    def export_npy(self, path: str, array=None) -> None:
        """
        Save a concentration array as a NumPy binary file.

        If array is omitted, the current/final field is exported. Passing
        initial_concentration exports the pre-diffusion starting field.
        """
        np.save(path, self.concentration if array is None else array)
        logger.info("Drug field NPY exported: %s", path)

    def export_csv(self, path: str, array=None) -> None:
        """
        Save a concentration array as a CSV file.
        """
        np.savetxt(path, self.concentration if array is None else array, delimiter=",", fmt="%.6f")
        logger.info("Drug field CSV exported: %s", path)

    def export_png(
        self,
        path: str,
        cmap=None,
        array=None,
        title: str = "Final antifungal field",
        vmax: float | None = None,
        colorbar_label: str = "Antifungal concentration",
    ) -> None:
        """
        Save a concentration array as a heatmap PNG.

        The default colour map is intentionally single-hue:

            no drug / 0 concentration  -> white
            high drug                  -> dark blue

        The optional vmax argument lets several simulations be plotted on the
        same colour scale. For example, set drug_plot_max_concentration = 8.0
        to make every heatmap use a 0..8xMIC range.
        """
        # Export either the live/final concentration grid or a supplied snapshot.
        C = self.concentration if array is None else array

        # Build the requested single-shade antifungal colour map if the caller
        # did not provide a custom Matplotlib colour map.
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list(
                "antifungal_white_to_dark_blue",
                ["#ffffff", "#08306b"],
            )

        # Fix the lower colour limit at zero because negative drug is invalid.
        # For the upper limit, use the user-requested value when provided;
        # otherwise auto-scale to the current array while avoiding a zero range.
        if vmax is None:
            finite_values = np.asarray(C)[np.isfinite(C)]
            vmax = max(float(np.max(finite_values)), 1e-9) if finite_values.size else 1.0

        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(
            C,
            origin="lower",
            extent=[self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max],
            aspect="equal",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info("Drug field PNG exported: %s", path)
