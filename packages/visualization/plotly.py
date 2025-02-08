import datetime
import numbers
from typing import List

import numpy as np
import pandas
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch


def histogram(
    df: pandas.DataFrame,
    title: str = "Default Title",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
):
    """
    Creates and displays an interactive histogram using Plotly, with hoverable information for each bin.

    Parameters:
        df (pandas.DataFrame): A DataFrame with two columns:
            - The first column contains names (used as hover text for bins).
            - The second column contains numerical data used to generate the histogram.
        title (str, optional): Title of the histogram. Defaults to "Default Title".
        x_label (str, optional): Label for the x-axis. Defaults to "X-Axis".
        y_label (str, optional): Label for the y-axis. Defaults to "Y-Axis".

    Raises:
        AssertionError: If some data points are not assigned to any bin.

    Notes:
        - The number of bins is fixed at 16.
        - Bin edges are calculated dynamically based on the range of the data.
        - Hover text shows the names of elements within each bin.
        - An annotation with the total number of elements and creation timestamp is added below the plot.
    """

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


def scatter(
    x: List[numbers.Number],
    y: List[numbers.Number],
    title: str = "Default Title",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    **kwargs,
):
    """
    Creates and displays a scatter plot using Plotly with customizable labels, title, and additional parameters.

    Parameters:
        x (List[numbers.Number]): A list of numerical values for the x-axis.
        y (List[numbers.Number]): A list of numerical values for the y-axis.
        title (str, optional): Title of the scatter plot. Defaults to "Default Title".
        x_label (str, optional): Label for the x-axis. Defaults to "X-Axis".
        y_label (str, optional): Label for the y-axis. Defaults to "Y-Axis".
        **kwargs: Additional keyword arguments passed to `go.Scatter` for customization (e.g., marker size, color).

    Notes:
        - The function uses `base_fig` to create the figure layout and adds a `Scatter` trace for the data.
        - The scatter plot is displayed immediately using `fig.show()`.
        - An annotation with the creation timestamp is added below the plot.
    """
    fig = base_fig(
        title=title,
        x_label=x_label,
        y_label=y_label,
        annotation_text=f"Created at: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')} ",
    )

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", **kwargs))

    fig.show()


def line_chart(
    x: List[numbers.Number],
    y: List[numbers.Number],
    title: str = "Default Title",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    **kwargs,
):
    """
    Creates and displays a scatter plot using Plotly with customizable labels, title, and additional parameters.

    Parameters:
        x (List[numbers.Number]): A list of numerical values for the x-axis.
        y (List[numbers.Number]): A list of numerical values for the y-axis.
        title (str, optional): Title of the scatter plot. Defaults to "Default Title".
        x_label (str, optional): Label for the x-axis. Defaults to "X-Axis".
        y_label (str, optional): Label for the y-axis. Defaults to "Y-Axis".
        **kwargs: Additional keyword arguments passed to `go.Scatter` for customization (e.g., marker size, color).

    Notes:
        - The function uses `base_fig` to create the figure layout and adds a `Scatter` trace for the data.
        - The scatter plot is displayed immediately using `fig.show()`.
        - An annotation with the creation timestamp is added below the plot.
    """
    fig = base_fig(
        title=title,
        x_label=x_label,
        y_label=y_label,
        annotation_text=f"Created at: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')} ",
    )

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", **kwargs))

    fig.show()


def base_fig(
    title: str = "Default Title",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    annotation_text: str = "",
) -> go.Figure:
    """
    Creates a base Plotly figure with a customizable title, axis labels, and annotation.

    Parameters:
        title (str, optional): Title of the figure. Defaults to "Default Title".
        x_label (str, optional): Label for the x-axis. Defaults to "X-Axis".
        y_label (str, optional): Label for the y-axis. Defaults to "Y-Axis".
        annotation_text (str, optional): Text to annotate below the plot. Defaults to an empty string.

    Returns:
        go.Figure: A Plotly Figure object with the specified layout and annotation.

    Notes:
        - The title and axis labels are automatically capitalized.
        - Annotations are positioned below the plot with no arrow and centered alignment.
        - This function creates a base figure, which can be further customized or populated with data.
    """

    fig = go.Figure()

    fig.update_layout(
        title=dict(text=title.title(), xanchor="center", yanchor="top", x=0.5),
        yaxis=dict(
            title=y_label.title(),
            ticklen=5,
            tickwidth=1,
            tickcolor="black",
            ticks="outside",
        ),
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
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.2,
        showarrow=False,
        align="center",
    )

    return fig


def boxplot(
    column: str,
    dfs: List[pd.DataFrame] = None,
) -> go.Figure:
    fig = base_fig(
        title=f"Boxplot of the {column.replace('_', ' ')}",
        x_label="Image Set",
        y_label=column.replace("_", " "),
    )

    for idx, df in enumerate(dfs):

        fig.add_trace(
            go.Box(
                y=df[column],
                text=df["picture_names"],
                boxpoints="all",
                hovertemplate="Picture: %{text} <extra></extra>",
                name=idx,
            )
        )

    return fig


def dendrogram(
    data: pandas.DataFrame,
    title: str = "Default Title",
):
    fig = ff.create_dendrogram(
        X=data, linkagefun=lambda x: sch.linkage(x, method="ward")
    )

    fig.update_layout(
        title=dict(text=title.title(), xanchor="center", yanchor="top", x=0.5),
    )

    fig.show()
