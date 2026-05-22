# gui/sim_gui.py

# Import tkinter GUI toolkit and its themed widgets
import tkinter as tk  
from tkinter import (
    ttk,        # Themed widget set
    Toplevel,   # Popup windows
    Label,      # Text display widgets
    Entry,      # Single-line text input
    Button,     # Clickable button
    Listbox,    # List selection widget
    END,        # Constant for Listbox end index
    SINGLE,     # Constant for single-selection mode
    filedialog  # File/cirectory chooser dialogs
)
import os
import random
import numpy as np # Imports for new unified seed

# Import threading primitives for running simulation in background
from threading import Thread  
# Time module for sleep during pause
import time  
# Matplotlib for embedding plots in the GUI
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  
# Core simulation setup and stepping functions
from core.options import Options  
from core.point import MPoint  
from main import step_simulation, setup_simulation, generate_outputs 
from io_utils.run_paths import resolve_seed, compute_run_dir # Per-run folder helpers


class OptionGUI:
    """Graphical interface to configure and run the mycelium simulator."""
    def __init__(self):
        # Create the main application window
        self.root = tk.Tk()
        # Set window title
        self.root.title("CyberMycelium - Simulator")

        # Instantiate default simulation options
        self.options = Options()
        # Dictionary to hold tk.Variable objects for each option field
        self.entries = {}

        # Thread object for running the simulation
        self.sim_thread = None
        # State flags for simulation control
        self.running = False
        self.paused = False
        # Placeholder for the Mycelium model and related components
        self.mycel = None
        self.components = {}

        # Variables for Run tab inputs
        self.max_steps_var = tk.StringVar(value="100")   # Max time steps
        self.max_tips_var = tk.StringVar(value="1000")   # Tip count limit
        self.output_folder = tk.StringVar(value="outputs")  # Output directory

        # Remember the computed per-run directory for this session
        self.current_run_dir = None

        # Matplotlib Figure and Axis for 3D plot
        self.fig = plt.Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = None  # Will hold the embedding of the Figure in Tk

        # Build all GUI components and start the event loop
        self.build_gui()
        self.root.mainloop()

    def build_gui(self):
        """Construct tabs, input fields, buttons, and plot canvas."""
        # Create a tabbed notebook widget
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # Define frames (tabs) for different option categories
        tabs = {
            "Core": ttk.Frame(notebook),
            "Branching": ttk.Frame(notebook),
            "Tropisms": ttk.Frame(notebook),
            "Density": ttk.Frame(notebook),
            "Nutrient": ttk.Frame(notebook),
            "Boundary": ttk.Frame(notebook),
            "RGB_Mutator": ttk.Frame(notebook),
            "Run": ttk.Frame(notebook)
        }
        # Add each frame to the notebook with its label
        for name, frame in tabs.items():
            notebook.add(frame, text=name)

        # CORE TAB 
        core_fields = [
            "growth_rate", "time_step",
            "length_scaled_growth", "length_growth_coef"
        ]
        core_frame = tabs["Core"]
        row = 0
        # For each core option, create a label + input widget
        for field in core_fields:
            if not hasattr(self.options, field):
                continue  # Skip if Options has no such attribute
            val = getattr(self.options, field)
            ttk.Label(core_frame, text=field).grid(column=0, row=row, sticky="w")
            if isinstance(val, bool):
                # Boolean options: use a checkbox
                var = tk.BooleanVar(value=val)
                ttk.Checkbutton(core_frame, variable=var).grid(column=1, row=row)
            else:
                # Numeric/string options: use a text entry
                var = tk.StringVar(value=str(val))
                ttk.Entry(core_frame, textvariable=var, width=18).grid(column=1, row=row)
            # Store the variable for later retrieval
            self.entries[field] = var
            row += 1

        # BRANCHING TAB
        branching_fields = [
            "branch_probability", "max_branches", "branch_angle_spread", "field_threshold",
            "branch_time_window", "branch_sensitivity", "optimise_initial_branching", "leading_branch_prob",
            "allow_internal_branching", "curvature_branch_bias", "branch_curvature_influence",
            "min_tip_age", "min_tip_length", "max_length", "max_age"
        ]
        branching_frame = tabs["Branching"]
        row = 0
        for field in branching_fields:
            if not hasattr(self.options, field):
                continue
            val = getattr(self.options, field)
            ttk.Label(branching_frame, text=field).grid(column=0, row=row, sticky="w")
            if isinstance(val, bool):
                var = tk.BooleanVar(value=val)
                ttk.Checkbutton(branching_frame, variable=var).grid(column=1, row=row)
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(branching_frame, textvariable=var, width=18).grid(column=1, row=row)
            self.entries[field] = var
            row += 1

        # TROPISMS TAB 
        tropism_fields = [
            "autotropism", "gravitropism", "random_walk",
            "gravi_angle_start", "gravi_angle_end", "gravi_angle_max", "gravi_layer_thickness",
            "anisotropy_enabled", "anisotropy_vector", "anisotropy_strength",
            "direction_memory_blend", "field_alignment_boost", "field_curvature_influence"
        ]
        tropism_frame = tabs["Tropisms"]
        row = 0
        for field in tropism_fields:
            if not hasattr(self.options, field):
                continue
            val = getattr(self.options, field)
            ttk.Label(tropism_frame, text=field).grid(column=0, row=row, sticky="w")
            if isinstance(val, bool):
                var = tk.BooleanVar(value=val)
                ttk.Checkbutton(tropism_frame, variable=var).grid(column=1, row=row)
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(tropism_frame, textvariable=var, width=18).grid(column=1, row=row)
            self.entries[field] = var
            row += 1

        # DENSITY TAB
        density_fields = [
            "die_if_old", "die_if_too_dense", "density_field_enabled", "density_threshold",
            "charge_unit_length", "neighbour_radius", "min_supported_tips",
            "density_from_tips", "density_from_branches", "density_from_all"
        ]
        density_frame = tabs["Density"]
        row = 0
        for field in density_fields:
            if not hasattr(self.options, field):
                continue
            val = getattr(self.options, field)
            ttk.Label(density_frame, text=field).grid(column=0, row=row, sticky="w")
            if isinstance(val, bool):
                var = tk.BooleanVar(value=val)
                ttk.Checkbutton(density_frame, variable=var).grid(column=1, row=row)
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(density_frame, textvariable=var, width=18).grid(column=1, row=row)
            self.entries[field] = var
            row += 1

        # NUTRIENT TAB
        nutrient_fields = [
            "use_nutrient_field",
            "nutrient_attraction", "nutrient_repulsion",
            "nutrient_radius", "nutrient_decay"
        ]
        nutrient_frame = tabs["Nutrient"]
        row = 0
        for field in nutrient_fields:
            if not hasattr(self.options, field):
                continue
            val = getattr(self.options, field)
            ttk.Label(nutrient_frame, text=field).grid(column=0, row=row, sticky="w")
            if isinstance(val, bool):
                var = tk.BooleanVar(value=val)
                ttk.Checkbutton(nutrient_frame, variable=var).grid(column=1, row=row)
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(nutrient_frame, textvariable=var, width=18).grid(column=1, row=row)
            self.entries[field] = var
            row += 1
        # Button to open detailed nutrient editor dialog
        Button(
            nutrient_frame,
            text="Nutrient Editor",
            command=self.open_nutrient_editor
        ).grid(column=0, row=row + 1, columnspan=2)

        # BOUNDARY TAB
        boundary_frame = tabs["Boundary"]
        row = 0
        # Checkbox for enabling volume constraint
        ttk.Label(boundary_frame, text="volume_constraint").grid(column=0, row=row, sticky="w")
        vc_var = tk.BooleanVar(value=self.options.volume_constraint)
        vc_chk = ttk.Checkbutton(boundary_frame, variable=vc_var)
        vc_chk.grid(column=1, row=row)
        self.entries["volume_constraint"] = vc_var
        row += 1

        # Entries for x_min, x_max, y_min, y_max, z_min, z_max
        bound_fields = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        for coord in bound_fields:
            ttk.Label(boundary_frame, text=coord).grid(column=0, row=row, sticky="w")
            val = getattr(self.options, coord)
            sv = tk.StringVar(value=str(val))
            entry = ttk.Entry(boundary_frame, textvariable=sv, width=12)
            entry.grid(column=1, row=row)
            # Disable if volume_constraint is unchecked
            if not self.options.volume_constraint:
                entry.config(state="disabled")
            # Store both the StringVar and the Entry widget for toggling later
            self.entries[coord] = (sv, entry)
            row += 1

        # Function to enable/disable boundary entries when checkbox changes
        def toggle_boundary_fields(*args):
            enabled = vc_var.get()
            for coord in bound_fields:
                sv, ent = self.entries[coord]
                ent.config(state="normal" if enabled else "disabled")

        # Trace checkbox variable to toggle fields automatically
        vc_var.trace_add("write", toggle_boundary_fields)
        toggle_boundary_fields()  # Initialize correct state

        # RGB MUTATOR TAB 
        color_frame = tabs["RGB_Mutator"]
        row = 0
        # Master toggle for enabling RGB mutations
        tk.Label(color_frame, text="Enable RGB Mutations").grid(column=0, row=row, sticky="w")
        rgb_var = tk.BooleanVar(value=self.options.rgb_mutations_enabled)
        ttk.Checkbutton(color_frame, variable=rgb_var).grid(column=1, row=row)
        self.entries["rgb_mutations_enabled"] = rgb_var
        row += 1

        # Initial RGB channels inputs
        for idx, channel in enumerate(("R", "G", "B")):
            ttk.Label(color_frame, text=f"initial_color_{channel}").grid(column=0, row=row, sticky="w")
            var = tk.StringVar(value=str(self.options.initial_color[idx]))
            ttk.Entry(color_frame, textvariable=var, width=10).grid(column=1, row=row)
            self.entries[f"initial_color_{channel.lower()}"] = var
            row += 1

        # Color mutation probability input
        ttk.Label(color_frame, text="color_mutation_prob").grid(column=0, row=row, sticky="w")
        prob_var = tk.StringVar(value=str(self.options.color_mutation_prob))
        ttk.Entry(color_frame, textvariable=prob_var, width=10).grid(column=1, row=row)
        self.entries["color_mutation_prob"] = prob_var
        row += 1

        # Color mutation scale input
        ttk.Label(color_frame, text="color_mutation_scale").grid(column=0, row=row, sticky="w")
        scale_var = tk.StringVar(value=str(self.options.color_mutation_scale))
        ttk.Entry(color_frame, textvariable=scale_var, width=10).grid(column=1, row=row)
        self.entries["color_mutation_scale"] = scale_var

        # RUN TAB 
        run_frame = tabs["Run"]
        # Max Steps label and entry
        ttk.Label(run_frame, text="Max Steps").grid(column=0, row=0, sticky="e")
        ttk.Entry(run_frame, textvariable=self.max_steps_var, width=10).grid(column=1, row=0)
        # Max Tips label and entry
        ttk.Label(run_frame, text="Max Tips").grid(column=0, row=1, sticky="e")
        ttk.Entry(run_frame, textvariable=self.max_tips_var, width=10).grid(column=1, row=1)
        # Output Folder label, entry, and Browse button
        ttk.Label(run_frame, text="Output Folder").grid(column=0, row=2, sticky="e")
        ttk.Entry(run_frame, textvariable=self.output_folder, width=20).grid(column=1, row=2)
        ttk.Button(run_frame, text="Browse", command=self.browse_folder).grid(column=2, row=2, padx=5)

        # Start and Pause/Resume buttons
        ttk.Button(run_frame, text="Start Simulation", command=self.start_sim).grid(column=0, row=3, columnspan=2, pady=8)
        ttk.Button(run_frame, text="Pause / Resume", command=self.toggle_pause).grid(column=0, row=4, columnspan=2)

        # Label to display metrics (step, tips, biomass)
        self.metrics_label = ttk.Label(run_frame, text="Step: 0 | Tips: 0 | Biomass: 0")
        self.metrics_label.grid(column=0, row=5, columnspan=2, pady=5)

        # Embed the Matplotlib Figure into the Tkinter GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=run_frame)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=8, padx=20, pady=10)

    def browse_folder(self):
        """Open folder chooser dialog and set the output_folder variable."""
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)

    def get_options(self):
        """
        Read current values from all entry widgets and update self.options accordingly.
        Returns:
            Options: populated Options dataclass.
        """
        # RGB-related keys handled separately later
        color_keys = {
            "rgb_mutations_enabled", "initial_color_r", "initial_color_g", "initial_color_b",
            "color_mutation_prob", "color_mutation_scale"
        }

        # Iterate over all stored entry variables
        for key, var in self.entries.items():
            if key in color_keys:
                continue  # Skip RGB fields here

            # volume_constraint is BooleanVar
            if key == "volume_constraint":
                self.options.volume_constraint = var.get()
                continue

            # Boundary fields: var is (StringVar, Entry widget)
            if key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
                str_var, entry_widget = var
                try:
                    val = float(str_var.get())
                    setattr(self.options, key, val)
                except ValueError:
                    pass  # Leave default if parse fails
                continue

            # Other fields: parse based on current type
            current_val = getattr(self.options, key)
            try:
                if isinstance(current_val, bool):
                    parsed = var.get() in ("1", "true", "True")
                elif isinstance(current_val, float) or "." in var.get():
                    parsed = float(var.get())
                elif isinstance(current_val, tuple):
                    parsed = tuple(map(float, var.get().strip("()").split(",")))
                else:
                    parsed = int(var.get())
            except Exception:
                parsed = current_val
            setattr(self.options, key, parsed)

        # Now parse RGB/mutation entries
        try:
            self.options.rgb_mutations_enabled = self.entries["rgb_mutations_enabled"].get()
            r = float(self.entries["initial_color_r"].get())
            g = float(self.entries["initial_color_g"].get())
            b = float(self.entries["initial_color_b"].get())
            self.options.initial_color = (r, g, b)
            self.options.color_mutation_prob = float(self.entries["color_mutation_prob"].get())
            self.options.color_mutation_scale = float(self.entries["color_mutation_scale"].get())
        except (KeyError, ValueError):
            print("⚠️ Invalid RGB/mutation parameters; using defaults.")

        return self.options

    def open_nutrient_editor(self):
        """
        Popup window allowing addition/removal of multiple nutrient attractors and repellents.
        """
        top = Toplevel(self.root)
        top.title("Nutrient Source Manager")

        # Labels and Listboxes for attractors and repellents
        Label(top, text="Nutrient Attractors").grid(row=0, column=0, columnspan=4)
        attr_listbox = Listbox(top, height=5, selectmode=SINGLE)
        attr_listbox.grid(row=1, column=0, columnspan=4, padx=5, pady=2, sticky="we")

        Label(top, text="Nutrient Repellents").grid(row=6, column=0, columnspan=4)
        rep_listbox = Listbox(top, height=5, selectmode=SINGLE)
        rep_listbox.grid(row=7, column=0, columnspan=4, padx=5, pady=2, sticky="we")

        # Inputs for new source coordinates and strength
        Label(top, text="X").grid(row=11, column=0)
        Label(top, text="Y").grid(row=11, column=1)
        Label(top, text="Z").grid(row=11, column=2)
        Label(top, text="Strength").grid(row=11, column=3)

        entry_x = Entry(top, width=6)
        entry_y = Entry(top, width=6)
        entry_z = Entry(top, width=6)
        entry_s = Entry(top, width=6)
        entry_x.grid(row=12, column=0)
        entry_y.grid(row=12, column=1)
        entry_z.grid(row=12, column=2)
        entry_s.grid(row=12, column=3)

        # Local copies of the options lists for editing
        attractors = self.options.nutrient_attractors[:]
        repellents = self.options.nutrient_repellents[:]

        def refresh_lists():
            """Update Listbox displays from attractors/repellents lists."""
            attr_listbox.delete(0, END)
            rep_listbox.delete(0, END)
            for pos, strength in attractors:
                attr_listbox.insert(END, f"{pos} : {strength}")
            for pos, strength in repellents:
                rep_listbox.insert(END, f"{pos} : {strength}")

        def add_entry(to_attr):
            """Add new entry from text fields to attractors or repellents."""
            try:
                x, y, z = float(entry_x.get()), float(entry_y.get()), float(entry_z.get())
                s = float(entry_s.get())
                (attractors if to_attr else repellents).append(((x, y, z), s))
                refresh_lists()
            except ValueError:
                print("⚠️ Invalid entry")

        def remove_entry(from_attr):
            """Remove selected entry from attractors or repellents."""
            lb = attr_listbox if from_attr else rep_listbox
            sel = lb.curselection()
            if sel:
                idx = sel[0]
                if from_attr:
                    del attractors[idx]
                else:
                    del repellents[idx]
                refresh_lists()

        # Buttons to add/remove entries
        Button(top, text="Add to Attractors", command=lambda: add_entry(True)).grid(row=13, column=0, columnspan=2)
        Button(top, text="Add to Repellents", command=lambda: add_entry(False)).grid(row=13, column=2, columnspan=2)
        Button(top, text="Remove Attractor", command=lambda: remove_entry(True)).grid(row=14, column=0, columnspan=2)
        Button(top, text="Remove Repellent", command=lambda: remove_entry(False)).grid(row=14, column=2, columnspan=2)

        def apply_and_close():
            """Commit local edits back to self.options and close dialog."""
            self.options.nutrient_attractors = attractors
            self.options.nutrient_repellents = repellents
            top.destroy()

        Button(top, text="✅ Apply & Close", command=apply_and_close).grid(row=15, column=0, columnspan=4, pady=5)
        refresh_lists()  # Initial population of listboxes

    def update_metrics_display(self, step):
        """
        Update the Step/Tips/Biomass label in the Run tab.
        Args:
            step (int): current simulation step.
        """
        tips = len(self.mycel.get_tips())
        if hasattr(self.mycel, "biomass_history") and self.mycel.biomass_history:
            biomass = self.mycel.biomass_history[-1]
        else:
            biomass = 0.0
        # Set new text on the metrics_label widget
        self.metrics_label.config(
            text=f"Step: {step} | Tips: {tips} | Biomass: {biomass:.2f}"
        )

    def draw_3d_mycelium(self):
        """Redraw the entire mycelium network in the embedded 3D plot."""
        self.ax.clear()
        self.ax.set_title("3D Mycelium Growth")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        for section in self.mycel.get_all_segments():
            # Extract start/end coordinates for plotting a line
            xs = [section.start.coords[0], section.end.coords[0]]
            ys = [section.start.coords[1], section.end.coords[1]]
            zs = [section.start.coords[2], section.end.coords[2]]
            self.ax.plot(xs, ys, zs, linewidth=1.0)
        # Refresh the canvas to show updates
        self.canvas.draw()

    def start_sim(self):
        """
        Callback for Start Simulation button.
        Reads options, sets up simulation, and spawns simulation thread.
        """
        if self.sim_thread and self.sim_thread.is_alive():
            print("Simulation already running.")
            return
        # Read GUI inputs into self.options
        opts = self.get_options()

        # Resolve seed, set RNGs, compute per-run output folder
        seed = resolve_seed(getattr(opts, "seed", None))
        opts.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Create run directory under chosen output root (GUI field)
        run_dir = compute_run_dir(self.output_folder.get(), seed)
        self.current_run_dir = run_dir
        # Inform downstream (main.py) to write all artefacts here
        os.environ["BATCH_OUTPUT_DIR"] = str(run_dir)
        print(f"📂 GUI run directory: {run_dir} (seed={seed})")
        
        # Initialise simulation model and components
        self.mycel, self.components = setup_simulation(opts)
        self.running = True
        self.paused = False
        # Start background thread for simulation loop
        self.sim_thread = Thread(target=self.run_simulation_loop, daemon=True)
        self.sim_thread.start()

    def toggle_pause(self):
        """
        Callback for Pause/Resume button.
        Toggles paused flag; when pausing, trigger immediate output generation.
        """
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            print("⏸️ Paused")
            # Generate outputs once when paused (write into current_run_dir)
            target_dir = str(self.current_run_dir or self.output_folder.get())
            self.root.after_idle(
                lambda: generate_outputs(
                    self.mycel,
                    self.components,
                    output_dir=target_dir
                )
            )
        else:
            print("▶️ Resuming")

    def run_simulation_loop(self):
        """
        Main simulation loop running in a separate thread.
        Advances simulation step-by-step, updates GUI, and handles termination.
        """
        # Parse max_steps and max_tips from StringVars
        try:
            max_steps = int(self.max_steps_var.get())
        except ValueError:
            max_steps = 100
        try:
            max_tips = int(self.max_tips_var.get())
        except ValueError:
            max_tips = 1000

        # Loop for each simulation step
        for step in range(max_steps):
            # If paused, sleep briefly until unpaused
            while self.paused:
                time.sleep(0.2)

            # Advance one step of the simulation
            step_simulation(self.mycel, self.components, step)
            # Update metrics label
            self.update_metrics_display(step)

            # Redraw the 3D plot every 3 steps
            if step % 3 == 0:
                self.draw_3d_mycelium()

            # Stop if tip count limit reached
            if len(self.mycel.get_tips()) >= max_tips:
                print(f"🛑 Max tips reached: {max_tips}")
                break

        # Mark as not running when loop ends
        self.running = False
        print("✅ Simulation complete")
        # Generate final outputs once on main thread
        target_dir = str(self.current_run_dir or self.output_folder.get())
        self.root.after_idle(
            lambda: generate_outputs(
                self.mycel,
                self.components,
                output_dir=target_dir
            )
        )
        # Final plot redraw
        self.draw_3d_mycelium()

# If this script is run directly, launch the GUI
if __name__ == "__main__":
    OptionGUI()
