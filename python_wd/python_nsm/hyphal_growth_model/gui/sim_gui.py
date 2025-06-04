# gui/sim_gui.py

import tkinter as tk
from tkinter import ttk, Toplevel, Label, Entry, Button, Listbox, END, SINGLE, filedialog
from threading import Thread
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.options import Options
from core.point import MPoint
from main import step_simulation, setup_simulation, generate_outputs

class OptionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CyberMycelium - Simulator")
        self.options = Options()
        self.entries = {}

        self.sim_thread = None
        self.running = False
        self.paused = False
        self.mycel = None
        self.components = {}

        self.max_steps_var = tk.StringVar(value="100")
        self.max_tips_var = tk.StringVar(value="1000")
        self.output_folder = tk.StringVar(value="outputs")  

        self.fig = plt.Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = None

        self.build_gui()
        self.root.mainloop()

    def build_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        tabs = {
            "Core": ttk.Frame(notebook),
            "Branching": ttk.Frame(notebook),
            "Tropisms": ttk.Frame(notebook),
            "Density": ttk.Frame(notebook),
            "Nutrient": ttk.Frame(notebook),
            "Boundary": ttk.Frame(notebook),
            "Run": ttk.Frame(notebook)
        }

        for name, tab in tabs.items():
            notebook.add(tab, text=name)

        field_categories = {
            "Core": ["growth_rate", "time_step", "length_scaled_growth", "length_growth_coef"],
            "Branching": ["branch_probability", "max_branches", "branch_angle_spread", "field_threshold",
                          "branch_time_window", "branch_sensitivity", "optimise_initial_branching", "leading_branch_prob",
                          "allow_internal_branching", "curvature_branch_bias", "branch_curvature_influence",
                          "min_tip_age", "min_tip_length", "max_length", "max_age"],
            "Tropisms": ["autotropism", "gravitropism", "random_walk", "gravi_angle_start", "gravi_angle_end",
                         "gravi_angle_max", "gravi_layer_thickness", "anisotropy_enabled", "anisotropy_vector",
                         "anisotropy_strength", "direction_memory_blend", "field_alignment_boost", "field_curvature_influence"],
            "Density": ["die_if_old", "die_if_too_dense", "density_field_enabled", "density_threshold",
                        "charge_unit_length", "neighbour_radius", "min_supported_tips", "density_from_tips",
                        "density_from_branches", "density_from_all"],
            "Nutrient": ["use_nutrient_field", "nutrient_attraction", "nutrient_repulsion",
                         "nutrient_radius", "nutrient_decay"],
            "Boundary": ["volume_constraint", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
        }

        for category, fields in field_categories.items():
            frame = tabs[category]
            row = 0
            for field in fields:
                if not hasattr(self.options, field):
                    continue
                val = getattr(self.options, field)
                ttk.Label(frame, text=field).grid(column=0, row=row, sticky="w")
                if isinstance(val, bool):
                    var = tk.BooleanVar(value=val)
                    ttk.Checkbutton(frame, variable=var).grid(column=1, row=row)
                else:
                    var = tk.StringVar(value=str(val))
                    ttk.Entry(frame, textvariable=var, width=18).grid(column=1, row=row)
                self.entries[field] = var
                row += 1

        # Nutrient popup
        Button(tabs["Nutrient"], text="🧪 Nutrient Editor", command=self.open_nutrient_editor).grid(column=0, row=row+1, columnspan=2)

        # Run tab
        sim_tab = tabs["Run"]
        ttk.Label(sim_tab, text="Max Steps").grid(column=0, row=0, sticky="e")
        ttk.Entry(sim_tab, textvariable=self.max_steps_var, width=10).grid(column=1, row=0)
        ttk.Label(sim_tab, text="Max Tips").grid(column=0, row=1, sticky="e")
        ttk.Entry(sim_tab, textvariable=self.max_tips_var, width=10).grid(column=1, row=1)

        # Output folder entry and browser
        ttk.Label(sim_tab, text="Output Folder").grid(column=0, row=2, sticky="e")
        ttk.Entry(sim_tab, textvariable=self.output_folder, width=20).grid(column=1, row=2)
        ttk.Button(sim_tab, text="Browse", command=self.browse_folder).grid(column=2, row=2, padx=5)

        ttk.Button(sim_tab, text="Start Simulation", command=self.start_sim).grid(column=0, row=3, columnspan=2, pady=8)
        ttk.Button(sim_tab, text="Pause / Resume", command=self.toggle_pause).grid(column=0, row=4, columnspan=2)

        self.metrics_label = ttk.Label(sim_tab, text="Step: 0 | Tips: 0 | Biomass: 0")
        self.metrics_label.grid(column=0, row=5, columnspan=2, pady=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=sim_tab)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=8, padx=20, pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)

    def get_options(self):
        for key, var in self.entries.items():
            val = getattr(self.options, key)
            try:
                if isinstance(val, bool):
                    parsed = var.get() in ("1", "true", "True")
                elif isinstance(val, float) or '.' in var.get():
                    parsed = float(var.get())
                elif isinstance(val, tuple):
                    parsed = tuple(map(float, var.get().strip("()").split(",")))
                else:
                    parsed = int(var.get())
            except Exception:
                parsed = val
            setattr(self.options, key, parsed)
        return self.options

    def open_nutrient_editor(self):
        top = Toplevel(self.root)
        top.title("Nutrient Source Manager")

        Label(top, text="Nutrient Attractors").grid(row=0, column=0, columnspan=4)
        attr_listbox = Listbox(top, height=5, selectmode=SINGLE)
        attr_listbox.grid(row=1, column=0, columnspan=4, padx=5, pady=2, sticky="we")

        Label(top, text="Nutrient Repellents").grid(row=6, column=0, columnspan=4)
        rep_listbox = Listbox(top, height=5, selectmode=SINGLE)
        rep_listbox.grid(row=7, column=0, columnspan=4, padx=5, pady=2, sticky="we")

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

        attractors = self.options.nutrient_attractors[:]
        repellents = self.options.nutrient_repellents[:]

        def refresh_lists():
            attr_listbox.delete(0, END)
            rep_listbox.delete(0, END)
            for pos, strength in attractors:
                attr_listbox.insert(END, f"{pos} : {strength}")
            for pos, strength in repellents:
                rep_listbox.insert(END, f"{pos} : {strength}")

        def add_entry(to_attr):
            try:
                x, y, z = float(entry_x.get()), float(entry_y.get()), float(entry_z.get())
                s = float(entry_s.get())
                (attractors if to_attr else repellents).append(((x, y, z), s))
                refresh_lists()
            except ValueError:
                print("⚠️ Invalid entry")

        def remove_entry(from_attr):
            lb = attr_listbox if from_attr else rep_listbox
            sel = lb.curselection()
            if sel:
                idx = sel[0]
                if from_attr:
                    del attractors[idx]
                else:
                    del repellents[idx]
                refresh_lists()

        Button(top, text="Add to Attractors", command=lambda: add_entry(True)).grid(row=13, column=0, columnspan=2)
        Button(top, text="Add to Repellents", command=lambda: add_entry(False)).grid(row=13, column=2, columnspan=2)
        Button(top, text="Remove Attractor", command=lambda: remove_entry(True)).grid(row=14, column=0, columnspan=2)
        Button(top, text="Remove Repellent", command=lambda: remove_entry(False)).grid(row=14, column=2, columnspan=2)

        def apply_and_close():
            self.options.nutrient_attractors = attractors
            self.options.nutrient_repellents = repellents
            top.destroy()

        Button(top, text="✅ Apply & Close", command=apply_and_close).grid(row=15, column=0, columnspan=4, pady=5)
        refresh_lists()

    def update_metrics_display(self, step):
        tips = len(self.mycel.get_tips())
        # pull last recorded biomass (if available)
        if hasattr(self.mycel, "biomass_history") and self.mycel.biomass_history:
            biomass = self.mycel.biomass_history[-1]
        else:
            biomass = 0.0
        # update the label to show biomass instead of “Total”
        self.metrics_label.config(
            text=f"Step: {step} | Tips: {tips} | Biomass: {biomass:.2f}"
        )        

    def draw_3d_mycelium(self):
        self.ax.clear()
        self.ax.set_title("3D Mycelium Growth")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        for section in self.mycel.get_all_segments():
            xs, ys, zs = zip(section.start.as_array(), section.end.as_array())
            self.ax.plot(xs, ys, zs, linewidth=1.0)
        self.canvas.draw()

    def start_sim(self):
        if self.sim_thread and self.sim_thread.is_alive():
            print("Simulation already running.")
            return
        opts = self.get_options()
        self.mycel, self.components = setup_simulation(opts)
        self.running = True
        self.paused = False
        self.sim_thread = Thread(target=self.run_simulation_loop, daemon=True)
        self.sim_thread.start()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            print("⏸️ Paused")
            self.root.after_idle(lambda: generate_outputs(self.mycel, self.components, output_dir=self.output_folder.get()))
        else:
            print("▶️ Resuming")

    def run_simulation_loop(self):
        try:
            max_steps = int(self.max_steps_var.get())
        except ValueError:
            max_steps = 100
        try:
            max_tips = int(self.max_tips_var.get())
        except ValueError:
            max_tips = 1000
        for step in range(max_steps):
            while self.paused:
                time.sleep(0.2)
            step_simulation(self.mycel, self.components, step)
            self.update_metrics_display(step)
            if step % 3 == 0:
                self.draw_3d_mycelium()
            if len(self.mycel.get_tips()) >= max_tips:
                print(f"🛑 Max tips reached: {max_tips}")
                break
        self.running = False
        print("✅ Simulation complete")
        self.root.after_idle(lambda: generate_outputs(self.mycel, self.components, output_dir=self.output_folder.get()))
        self.draw_3d_mycelium()

if __name__ == "__main__":
    OptionGUI()
