import datetime

import numpy as np
import pandas
import plotly.graph_objects as go


def histogram(
    df: pandas.DataFrame,
    title: str = "Default Title",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
):

    data = df.iloc[:, 1]
    names = df.iloc[:, 0]
    num_bins = 16

    bins = np.linspace(min(data), max(data) + 1e-5, num_bins + 1)

    # Subtraction by -1 is necessary, because np.digitize is sorting from 1 to n and np.bincount counts from 0 to n
    bin_indices = np.digitize(data, bins) - 1

    elements_per_bin = [[] for _ in range(1, len(bins))]

    for name, index in zip(names, bin_indices):
        elements_per_bin[index].append(name)

    number_of_elements_per_bin = np.bincount(bin_indices)

    if sum(number_of_elements_per_bin) < len(data):
        raise AssertionError("Not all datapoints are distributed to a bin.")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure(
        go.Bar(
            x=bin_centers,
            y=number_of_elements_per_bin,
            hovertext=[
                "<br>".join([f"• {text}" for text in images_per_bin])
                for images_per_bin in elements_per_bin
            ],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=dict(text=title.title(), xanchor="center", yanchor="top", x=0.5),
        yaxis=dict(
            title=y_label.title(),
            ticklen=5,
            tickwidth=1,
            tickcolor="black",
            ticks="outside",
        ),
        bargap=0.1,
        xaxis=dict(
            title=x_label.title(),
            ticklen=5,
            title_standoff=2,
            tickwidth=1,
            tickcolor="black",
            ticks="outside",
        ),
    )

    fig.add_annotation(
        text=f"Number of elements: {len(data)} | Created at: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')} ",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.2,
        showarrow=False,
        align="center",
    )

    fig.show()
