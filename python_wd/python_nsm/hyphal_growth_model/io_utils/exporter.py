# io_utils/exporter.py

from core.mycel import Mycel
from core.section import Section
import csv

def export_to_csv(mycel: Mycel, filename="mycelium.csv", all_time=False):
    import csv

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        if all_time:
            writer.writerow(["step", "tip_index", "x", "y", "z", "age", "length"])
            for step_idx, snapshot in enumerate(mycel.time_series):
                for i, tip in enumerate(snapshot):
                    row = [
                        step_idx,
                        i,
                        tip["x"],
                        tip["y"],
                        tip["z"],
                        tip["age"],
                        tip["length"]
                    ]
                    writer.writerow(row)
        else:
            writer.writerow(["x0", "y0", "z0", "x1", "y1", "z1", "length", "age", "is_tip", "is_dead"])
            for s in mycel.get_all_segments():
                row = list(s.start.coords) + list(s.end.coords)
                row += [s.length, s.age, s.is_tip, s.is_dead]
                writer.writerow(row)

    print(f"📄 CSV exported: {filename}")

def export_to_obj(mycel: Mycel, filename="mycelium.obj"):
    vertices = []
    edges = []

    # Write each section as two vertices and one edge
    for i, s in enumerate(mycel.get_all_segments()):
        v_start = s.start.coords
        v_end = s.end.coords
        vertices.append(v_start)
        vertices.append(v_end)
        edges.append((2*i+1, 2*i+2))  

    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for e in edges:
            f.write(f"l {e[0]} {e[1]}\n")
    print(f"🌐 OBJ exported: {filename}")

def export_tip_history(mycel, filename="mycelium_time_series.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z"])  

        for time, tips in mycel.step_history:
            for x, y, z in tips:
                writer.writerow([f"{time:.2f}", x, y, z])

    print(f"🧪 Tip position time series exported: {filename}")

def export_biomass_history(mycel: Mycel, filepath: str):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "biomass"])
        for step_index, biomass in enumerate(mycel.biomass_history):
            t = step_index * mycel.options.time_step
            writer.writerow([t, biomass])
    print(f"🪵 Biomass history at each time step exported: {filename}")
