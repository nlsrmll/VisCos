from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


def histogram(data: List[Tuple[np.floating, str]], title:str = "Default Title", x_label:str="X-Axis", y_label:str="Y-Axis"):

    num_bins = 16

    bin_size = (max(data) - min(data)) / num_bins

    fig = go.Figure(go.Histogram(x=data, xbins=dict(start=min(data), end=max(data), size=bin_size),hoverinfo="x+y" ))
    fig.update_layout(bargap=0.2, title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()

def histogram_as_bar(data: List[float | np.floating], title:str = "Default Title", x_label:str="X-Axis", y_label:str="Y-Axis"):


    num_bins = 16

    bins = np.linspace(min(data), max(data) + 1e-5, num_bins+1)

    #Substraction by -1 is necessary, because np.digitize is sorting from 1 to n and np.bincount counts from 0 to n
    bin_indices = np.digitize(data, bins) -1

    number_of_elements_per_bin = np.bincount(bin_indices)

    if sum(number_of_elements_per_bin) < len(data):
        raise AssertionError("Not all datapoints are distributed to a bin.")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure(go.Bar(x=bin_centers, y=number_of_elements_per_bin))
    fig.update_layout(
        title=dict(
            text=title,
            xanchor="center",
            yanchor="top",
            x=0.5
        ),
        yaxis=dict(
            title=y_label,
            ticklen=5,
            tickwidth=1,
            tickcolor='black',
            ticks='outside'
        ),
        bargap=0.1,
        xaxis=dict(
            title=x_label,
            ticklen=5,
            tickwidth=1,
            tickcolor='black',
            ticks='outside'
        )
    )

    fig.show()

