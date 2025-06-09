# vis/plotly_3d_export.py

import plotly.graph_objs as go
import os
from core.mycel import Mycel

def plot_mycel_3d_interactive(mycel: Mycel, save_path="outputs/mycelium_3d_interactive.html"):
    traces = []
    # helper to convert float RGB to hex color string
    def to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )    

    for section in mycel.get_all_segments():
        for start, end in section.get_subsegments():
            xs, ys, zs = zip(start.coords, end.coords)
            trace = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(width=2, color=to_hex(section.color)),
                showlegend=False
            )            
            traces.append(trace)

        if section.is_tip and not section.is_dead:
            x, y, z = section.end.coords
            tip_marker = go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=4, color=to_hex(section.color)),
                name='Tip'
            )
            traces.append(tip_marker)

    layout = go.Layout(
        title='Interactive 3D Mycelium',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=traces, layout=layout)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"🌐 Interactive 3D plot saved to: {save_path}")
