import os
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from PIL import Image
from models.file import File
from packages.study_data.analyzer import StudyAnalyzer
from packages.utils.colors import generate_color_palett
from packages.utils.io import get_full_image_name
from packages.utils.plot_utils import plot_images
from packages.utils.settings import settings
from packages.visualization.plotly import base_fig, histogram, base_subplot


class StudyDataVisualizer:

    def __init__(self, analyzer: StudyAnalyzer):
        self._analyzer = analyzer

        self.__images_in_dir: list[File] = [
            File(
                filename=os.path.splitext(singleFile)[0],
                extension=os.path.splitext(singleFile)[1].lower(),
            )
            for singleFile in os.listdir(settings.IMAGE_BASE_PATH)
        ]

    # TODO: Im namen anzeigen, wann die PCA gemacht wurde
    def show_raw_data(self, dim_reduction_method: Literal["pca", "tsne", "umap"]):
        fig = base_fig()
        column = "PCA"
        if dim_reduction_method == "tsne":
            column = "t_SNE"

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=self._analyzer.data[f"{column}_D1"],
                y=self._analyzer.data[f"{column}_D2"],
            )
        )

        fig.show()

    def cluster_as_scatter(
        self, clustering: Literal["kMean", "hierarchical"] = "kMean"
    ):

        if self._analyzer.pca is None:
            raise ValueError("PCA is not available.")

        columns = [f"PCA_D{idx +1}" for idx in range(self._analyzer.pca.n_components)]
        columns.append(f"{clustering}_labels")
        groups = self._analyzer.data[columns].groupby(f"{clustering}_labels")

        cluster_centers = (
            self._analyzer.data[columns].groupby(f"{clustering}_labels").mean()
        )

        fig = base_fig(title=f"{clustering} Clustering".title())

        if self._analyzer.pca.n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=cluster_centers["PCA_D1"],
                    y=cluster_centers["PCA_D2"],
                    mode="markers",
                    marker=dict(color="black", symbol="x", size=12),
                    name="Centroids",
                    hovertemplate="%{text} <extra></extra>",
                    text=[
                        f"Center of Cluster {cluster}"
                        for cluster in cluster_centers.index.tolist()
                    ],
                )
            )
            colors = generate_color_palett(self._analyzer.cluster_count)
            for idx, (group_name, df) in enumerate(groups):
                fig.add_trace(
                    go.Scatter(
                        x=df["PCA_D1"],
                        y=df["PCA_D2"],
                        mode="markers",
                        marker=dict(color=colors[idx], size=8),
                        name=f"Cluster {group_name}",
                        hovertemplate="%{text} <extra></extra>",
                        text=[
                            f"Participant: {participant_id}"
                            for participant_id in df.index.tolist()
                        ],
                    )
                )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_centers["PCA_D1"],
                    y=cluster_centers["PCA_D2"],
                    z=cluster_centers["PCA_D3"],
                    mode="markers",
                    marker=dict(color="black", symbol="x", size=4),
                    name="Centroids",
                    hovertemplate="%{text} <extra></extra>",
                    text=[
                        f"Center of Cluster {cluster}"
                        for cluster in cluster_centers.index.tolist()
                    ],
                )
            )
            colors = generate_color_palett(self._analyzer.cluster_count)
            for idx, (group_name, df) in enumerate(groups):
                fig.add_trace(
                    go.Scatter3d(
                        x=df["PCA_D1"],
                        y=df["PCA_D2"],
                        z=df["PCA_D3"],
                        mode="markers",
                        marker=dict(color=colors[idx], size=4),
                        name=f"Cluster {group_name}",
                        hovertemplate="%{text} <extra></extra>",
                        text=[
                            f"Participant: {participant_id}"
                            for participant_id in df.index.tolist()
                        ],
                    )
                )

        fig.show()

    def cluster_as_boxplot(self, of_pca_component: int = 1):

        if self._analyzer.pca is None:
            raise ValueError("PCA is not available.")

        if of_pca_component > self._analyzer.pca.n_components:
            raise IndexError(
                f"There are only {self._analyzer.pca.n_components} components. You are trying to access the {of_pca_component}. component"
            )

        groups = self._analyzer.data[[f"PCA_D{of_pca_component}", "cluster"]].groupby(
            "cluster"
        )
        colors = generate_color_palett(self._analyzer.cluster_count)
        fig = base_fig(title="Boxplot of all clusters", x_label="Cluster Number")

        for idx, (group_name, df) in enumerate(groups):
            fig.add_trace(
                go.Box(
                    y=df[f"PCA_D{of_pca_component}"],
                    name=f"Cluster {group_name}",
                    text=df.index.tolist(),
                    boxpoints="all",
                    hovertemplate="Participant: %{text} <extra></extra>",
                    marker=dict(color=colors[idx]),
                )
            )

        fig.show()

    def pca_most_influential_images(self, image_number: int = 1, pca_index: int = 0):
        if self._analyzer.pca is None:
            raise ValueError("PCA is not available.")

        if pca_index > self._analyzer.pca.n_components:
            raise IndexError(
                f"There are only {self._analyzer.pca.n_components} components. You are trying to access the {pca_index}. component"
            )

        influential_indices = np.argpartition(
            np.abs(self._analyzer.pca.components_[pca_index]), -image_number
        )[-image_number:]

        most_influential_images = [
            {
                "data": Image.open(
                    os.path.join(settings.IMAGE_BASE_PATH, get_full_image_name(idx))
                ).convert("RGB"),
                "image_name": get_full_image_name(idx),
                "component_value": self._analyzer.pca.components_[pca_index][idx],
            }
            for idx in influential_indices
        ]

        plot_images(
            most_influential_images[::-1],
            "Meist einflussreiche Bilder (PCA-Komponente)",
        )

    def pca_least_influential_images(self, image_number: int = 1, pca_index: int = 0):
        if self._analyzer.pca is None:
            raise ValueError("PCA is not available.")

        if pca_index > self._analyzer.pca.n_components:
            raise IndexError(
                f"There are only {self._analyzer.pca.n_components} components. You are trying to access the {pca_index}. component"
            )

        influential_indices = np.argpartition(
            np.abs(self._analyzer.pca.components_[pca_index]), image_number
        )[:image_number]

        most_influential_images = [
            {
                "data": Image.open(
                    os.path.join(settings.IMAGE_BASE_PATH, get_full_image_name(idx))
                ).convert("RGB"),
                "image_name": get_full_image_name(idx),
                "component_value": self._analyzer.pca.components_[pca_index][idx],
            }
            for idx in influential_indices
        ]

        plot_images(
            most_influential_images,
            "Geringst einflussreiche Bilder (PCA-Komponente)",
        )

    def show_cluster_voting_for_picture(self, picture_number: int):

        grouped_clusters = self._analyzer.data.groupby("kMean_labels")

        fig = base_fig(
            title=f"Cluster Voting for Picture {picture_number:05}",
            x_label=f"Cluster Number",
            y_label="Cluster Mean Voting",
        )

        for cluster_id, group in grouped_clusters:
            picture_voting = group[picture_number]

            if picture_voting is None:
                raise KeyError(
                    f"Could not find the picture with the ID {picture_number} in the Dataset."
                )

            cluster_mean = picture_voting.mean()
            cluster_std = picture_voting.std()

            fig.add_trace(
                go.Bar(
                    x=[cluster_id],
                    y=[cluster_mean],
                    name=f"Cluster {cluster_id}",
                    error_y=dict(type="data", array=[cluster_std]),
                )
            )

        fig.show()

    def show_cluster_votings(self):
        grouped_clusters = self._analyzer.data.groupby("kMean_labels")
        picture_column_names = self._analyzer.get_original_columns()
        fig = base_fig()

        traces = []
        for picture_idx, picture_name in enumerate(picture_column_names):
            visibility = [False] * len(picture_column_names)
            visibility[picture_idx] = True

            for cluster_id, group in grouped_clusters:
                cluster_mean = group[picture_idx].mean()
                cluster_std = group[picture_idx].std()

                traces.append(
                    go.Bar(
                        x=[f"{get_full_image_name(picture_name)}"],
                        y=[cluster_mean],
                        name=f"Cluster {cluster_id}",
                        error_y=dict(type="data", array=[cluster_std]),
                        visible=(picture_idx == 0),
                    )
                )

        steps = []

        for idx, picture_name in enumerate(picture_column_names):
            full_image_name = get_full_image_name(picture_name)
            step = dict(
                method="update",
                args=[
                    {"visible": [full_image_name in t.x for t in traces]},
                    {"title": f"Cluster Voting for {full_image_name}"},
                ],
                label=full_image_name,
            )
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Picture: "},
                pad={"t": len(picture_column_names)},
                steps=steps,
            )
        ]

        fig.add_traces(traces)
        fig.update_layout(
            sliders=sliders,
            barmode="group",
        )
        fig.show()
