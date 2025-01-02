import math
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure


def histogram(data: List[float | np.floating], title:str = "Default Title", x_label:str="X-Axis", y_label:str="Y-Axis") -> Figure:

    num_bins = 16

    #Increasing the upper border by an epsilon to include the last datapoint in the last bin
    bins = np.linspace(start=min(data), stop=max(data) + 1e-5, num=num_bins)

    bin_indices = np.digitize(data, bins)

    bin_counts = [bin_indices.tolist().count(i) for i in range(1,len(bins))]

    if sum(bin_counts) < len(data):
        raise Exception("Not all datapoints where distributed in bins.")

    bin_elements = {i:[] for i in range(len(bins))}

    for d,b in zip(data, bin_indices):
        bin_elements[b].append(d)

    # Texte für Hover-Info erstellen
    bin_hover_text = [
        f"Bin {i}: {bin_elements[i]}" for i in range(1, len(bins))
    ]

    min_value = math.floor(min(data) / 10) * 10
    max_value = math.ceil(max(data)/10) * 10



    print(min_value, max_value)
    # Plotly-Histogramm erstellen
    fig = go.Figure(go.Bar(
        y=bin_counts,  # Häufigkeiten
        text=bin_hover_text,  # Hover-Text
        hoverinfo="x+y+text"  # Zeige x-Wert, y-Wert und Text
    ))

    fig.update_layout(
        title="Histogram - Contrast",
        yaxis_title="Number of Pictures",
        xaxis=dict(
            title="Contrast",
            tickvals=np.linspace(min_value, max_value, num=10),
            ticktext=[f"{int(val)}" for val in np.linspace(min_value, max_value, num=10)],
        )
    )

    fig.show()

