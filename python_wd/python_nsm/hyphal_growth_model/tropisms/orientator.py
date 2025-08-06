# tropisms/orientator.py

# Imports
from core.point import MPoint # 3D point/vector ops
from core.section import Section # Section segments in mycelium
from core.options import Options # Sim params
from compute.field_aggregator import FieldAggregator # Aggregated multiple field sources
from vis.anisotropy_grid import AnisotropyGrid # Grid-based anisotropu directions
import numpy as np # Numerical utilities

class Orientator:
    """Combines multiple tropism influences to compute new growth orientation."""
    def __init__(self, options: Options):
        self.options = options # Store global sim opts
        self.aggregator: FieldAggregator = None # Placeholder for FieldAggregator instance
        self.density_grid = None # Placeholder for density grid (avoidance)
        self.anisotropy_grid: AnisotropyGrid = None # Placeholder for anisotropy grid
        self.nutrient_sources: list[MPoint] = [] # List of nutrient source points (MPoint instances)

    def set_field_source(self, aggregator: FieldAggregator):
        self.aggregator = aggregator # Assign the FieldAggregator for chemical/substrate fields

    def set_density_grid(self, grid):
        self.density_grid = grid # Assign a density grid object that provides avoidance gradients

    def set_anisotropy_grid(self, grid: AnisotropyGrid):
        self.anisotropy_grid = grid # Assign an anisotropy grid for directional bias

    def set_nutrient_sources(self, points: list[MPoint]):
        self.nutrient_sources = points # Set list of nutrient attractor/repellent points

    def compute(self, section: Section) -> MPoint:
        """ 
        Compute the new orientation vector for a given Section,
        combining autotropism, external fields, density avoidance,
        gravitropism, nutrient ces, anisotropu, randomness, and directional memory.
        """
        orientation = section.orientation.copy() # Start w/ current orientation as a mutable copy

        # Autotropism & Field Alignment
        grad = None # grad Placeholder
        
        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end) # Compute scalar field and gradeint at section end
            if grad is not None:
                orientation.add(grad.scale(self.options.autotropism)) # Push orientation slightly along gradient direction

                # Boost alignment with field gradient
                if self.options.field_alignment_boost > 0: # If enabled, amplify component along gradient
                    grad_unit = grad.copy().normalise() # Unit gradient vector
                    dot = orientation.dot(grad_unit) # Dot product = cos(angle) between orientation & gradient
                    if dot > 0:
                        boost = grad_unit.scale(dot * self.options.field_alignment_boost) # Boost vector proportional to dot*boost factor
                        orientation.add(boost)
                        print(f"Gradient alignment boost applied: dot={dot:.2f}, boost={boost}")

                # Curvature influence from field
                if self.options.field_curvature_influence > 0: # If enabled, bend toward field curvature
                    curvature = self.aggregator.compute_field_curvature(section.end) # Approximate Laplacian (curvature) of scalar field
                    direction = grad.copy().normalise() # Unit direction of gradient
                    orientation.add(direction.scale(curvature * self.options.field_curvature_influence)) # Scale curvature by influence factor an add
                    print(f"🌀 Curvature contribution: value={curvature:.3f}, scaled={curvature * self.options.field_curvature_influence:.3f}")

        # Density-based avoidance
        if self.options.die_if_too_dense and self.density_grid:
            density_grad = self.density_grid.get_gradient_at(section.end) # Get local density gradient (points toward higher density)
            orientation.subtract(density_grad) # Subtract to steer away from high-density regions

        # Gravitropism
        if self.options.gravitropism > 0:
            z = section.end.coords[2] # Current height (Z-coord)
            if z < self.options.gravi_angle_start: # Determine gravity influence strength via linear falloff
                strength = 0
            elif z > self.options.gravi_angle_end:
                strength = self.options.gravitropism
            else:
                # Interpolar between start and end heights
                t = (z - self.options.gravi_angle_start) / (self.options.gravi_angle_end - self.options.gravi_angle_start)
                strength = t * self.options.gravitropism
            gravity_vec = MPoint(0, -1, 0).scale(strength) # Gravity vector points downward in Y (negative Y-axis)
            orientation.add(gravity_vec)

        # Nutrient fields
        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end) # Vector from section end toward nutrient source
            dist = delta.magnitude() # Distance to nutrient

            if dist < self.options.nutrient_radius: # Only consider if within specified radius
                influence = 1.0 - (dist / self.options.nutrient_radius) # Influence decays linearly w/ distance

                if self.options.nutrient_attraction > 0: # Attraction component
                    orientation.add(
                        delta.copy().normalise().scale(self.options.nutrient_attraction * influence)
                    )
                if self.options.nutrient_repulsion > 0: # Repulsion component
                    orientation.subtract(
                        delta.copy().normalise().scale(self.options.nutrient_repulsion * influence)
                    )

        # Global or Grid-Based Anisotropy
        if self.options.anisotropy_enabled:
            if self.anisotropy_grid:
                dir_vec = self.anisotropy_grid.get_direction_at(section.end) # Grid-based anisotropu direction
                print(f"🧭 Grid-based anisotropy: {dir_vec}")
            else:
                dir_vec = MPoint(*self.options.anisotropy_vector).normalise() # Global constant anisotropy vector from opts
                print(f"🧭 Global anisotropy: {self.options.anisotropy_vector}")

            orientation.add(dir_vec.scale(self.options.anisotropy_strength)) # Add anisotropy influence

        # Random walk
        if self.options.random_walk > 0:
            rand = np.random.normal(0, 1, 3) # Draw 3 independent Gaussian samples
            # Noamlise random direction and scale by random_walk magnitude
            orientation.add(MPoint(*rand).normalise().scale(self.options.random_walk))

        # Directional memory blending 
        if self.options.direction_memory_blend > 0 and section.orientation:
            blend = self.options.direction_memory_blend
            # Combine old orientation and new proposal via EMA
            orientation = (
                section.orientation.copy().scale(blend)
                .add(orientation.scale(1.0 - blend))
                .normalise()
            )
            print(f"🧠 Orientation memory applied: blend={blend:.2f}")

        return orientation.normalise() # Ensure final orientation is a unit vector
