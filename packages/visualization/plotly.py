import math
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure


def histogram(data: List[Tuple[np.floating, str]], title:str = "Default Title", x_label:str="X-Axis", y_label:str="Y-Axis"):

    num_bins = 16

    bin_size = (max(data) - min(data)) / num_bins

    fig = go.Figure(go.Histogram(x=data, xbins=dict(start=min(data), end=max(data), size=bin_size),hoverinfo="x+y" ))
    fig.update_layout(bargap=0.2, title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()

