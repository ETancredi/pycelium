# vis/plotly_3d_export.py

# Imports
import plotly.graph_objs as go # 3D interactive plots
import os # Filesystem os
from core.mycel import Mycel # Mycel class for sim data

def plot_mycel_3d_interactive(mycel: Mycel, save_path="outputs/mycelium_3d_interactive.html"):
    traces = [] # accumulate plotly trace objects here
    # helper to convert float RGB to hex color string
    def to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )    

    for section in mycel.get_all_segments(): # Iterate over every segment in mycelium network
        for start, end in section.get_subsegments(): # Draw each stored subsegment as a 3D line
            xs, ys, zs = zip(start.coords, end.coords) # Unpack start and end coords
            trace = go.Scatter3d(
                x=xs, y=ys, z=zs, # coords of segment ends
                mode='lines', # render as a line
                line=dict(width=2, color=to_hex(section.color)), # thickness and colour
                showlegend=False # no legend
            )            
            traces.append(trace) # Add segment trace to list

        if section.is_tip and not section.is_dead: # Makr the tip of each living segment with a scatter point
            x, y, z = section.end.coords # coords of tip end
            tip_marker = go.Scatter3d(
                x=[x], y=[y], z=[z], # single points on x, y and z
                mode='markers', # render as markers
                marker=dict(size=4, color=to_hex(section.color)), # marker size and colour
                name='Tip' # label for legend (not shown)
            )
            traces.append(tip_marker) # Add tip marker trace

    layout = go.Layout( # Define layout for 3D figure
        title='Interactive 3D Mycelium', # Figure title
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), # Lables for x, y, z-axes
        margin=dict(l=0, r=0, b=0, t=40) # tight margins
    )
    fig = go.Figure(data=traces, layout=layout) # Create Plotly figure from traces and layout

    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure output directory exists before writing file
    fig.write_html(save_path) # Write interactive HTML file
    print(f"🌐 Interactive 3D plot saved to: {save_path}")
