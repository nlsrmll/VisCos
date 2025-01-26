import colorsys
import random
from typing import List, Literal, Tuple
from PIL import Image
import pandas as pd
import plotly.express
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

from packages.clustering.model.ImageAnalyzer import ImageAnalyzer
from packages.visualization.plotly import base_fig, scatter
from matplotlib import pyplot as plt


def generate_colors(color_count: int) -> List[str]:
    """
    Generates a list of distinct colors in RGB format, evenly distributed across the HSV color space.

    Parameters:
        color_count (int): The number of distinct colors to generate.

    Returns:
        list: A list of color strings in the format "rgb(r, g, b)", where `r`, `g`, and `b` are values between 0 and 255.

    Notes:
        - Colors are generated using the HSV color model and then converted to RGB.
        - The `hue` value is evenly distributed across the range [0, 1] for `color_count` distinct colors.
    """
    colors = []
    for i in range(color_count):
        hue = i / color_count
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 0.85)
        colors.append(f"rgb({rgb[0]*255}, {rgb[1]*255}, {rgb[2]*255})")
    return colors


class ClusterAnalyze:

    def __init__(self, data: pd.DataFrame, seed: int = None, cluster_count: int = None):
        self.data = data
        # TODO: Kann man hier nicht einfach die Columnspalten speichern,
        # sodass man nur diese aus dem DF herausnimmt? Spart speicher
        self.original_data = data.copy()
        self.seed = seed
        self.cluster_count = cluster_count
        self.image_analysis: List[ImageAnalyzer] = []

    def visualize_elbow_curve(self, cluster_count: int) -> None:
        """
        Visualizes the elbow curve to determine the optimal number of clusters for K-Means clustering.

        Parameters:
            cluster_count (int): The maximum number of clusters to evaluate.

        Notes:
            - The function computes the elbow curve using `calculate_elbow_curve` and visualizes it as a scatter plot.
            - The x-axis represents the number of clusters, and the y-axis represents the inertia (sum of squared distances
              of samples to their nearest cluster center).
            - The "elbow" point in the curve suggests the optimal number of clusters for the dataset.
        """
        means, inertias = self.calculate_elbow_curve(cluster_count)

        scatter(
            means,
            inertias,
            title="Elbow Curve for K-Nearest Neighbors",
            x_label="Cluster",
            y_label="Inertia",
        )

    def calculate_elbow_curve(
        self, cluster_count: int
    ) -> Tuple[List[int], List[float]]:
        """
        Calculates the elbow curve for determining the optimal number of clusters in K-Means clustering.

        Parameters:
            cluster_count (int): The maximum number of clusters to evaluate.

        Returns:
            Tuple[List[int], List[float]]:
                - A list of cluster counts.
                - A list of corresponding inertia values (sum of squared distances of samples to their nearest cluster center).

        Notes:
            - The function iteratively fits K-Means for cluster counts ranging from 1 to `cluster_count`.
            - The inertia values indicate how well the data is clustered. The "elbow" point in the curve can suggest
              the optimal number of clusters.
            - The `random_state` for reproducibility is determined by `self.seed` or a randomly generated value.
        """
        means = []
        inertias = []

        for cluster_count in range(1, cluster_count + 1):
            k_means = KMeans(
                n_clusters=cluster_count,
                random_state=(
                    self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
                ),
            )

            k_means.fit(self.data)

            means.append(cluster_count)
            inertias.append(k_means.inertia_)

        return means, inertias

    def calculate_knn(
        self,
        cluster_count: int = None,
        pca: bool = True,
    ) -> List[float]:
        """
        Calculates K-Nearest Neighbors (KNN) clusters using the K-Means algorithm, with optional PCA dimensionality reduction.

        Parameters:
            cluster_count (int): The number of clusters to compute.
            pca (bool, optional): If True, applies PCA to reduce data to 2 dimensions before clustering. Defaults to False.

        Returns:
            List[float]: A list of cluster labels for each data point.

        Notes:
            - The `KMeans` algorithm is used for clustering, with the number of clusters specified by `cluster_count`.
            - If `pca` is True, the dataset is reduced to 2 principal components using PCA, and the dimensions
              are added to the dataset as `PCA_D1` and `PCA_D2`.
            - The `random_state` for reproducibility is determined by `self.seed` or a randomly generated value.
        """
        if self.cluster_count is None and cluster_count is None:
            raise ValueError()
        if self.cluster_count is None:
            self.cluster_count = cluster_count

        k_means = KMeans(
            n_clusters=cluster_count,
            random_state=(
                self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
            ),
        )

        if pca:
            data_pca = PCA(n_components=2).fit_transform(self.data)
            self.data["PCA_D1"] = data_pca[:, 0]
            self.data["PCA_D2"] = data_pca[:, 1]

            labels = k_means.fit_predict(self.data[["PCA_D1", "PCA_D2"]])
        else:
            labels = k_means.fit_predict(self.data)

        self.data["cluster"] = labels

        return labels

    def visualize_knn(
        self,
        cluster_count: int,
        pca_timing: Literal["before", "after"] = "before",
    ) -> None:
        """
        Visualizes K-Nearest Neighbors (KNN) clusters with optional PCA dimensionality reduction.

        Parameters:
            cluster_count (int): The number of clusters to compute and visualize.
            pca_timing (Literal["before", "after"], optional): Determines when PCA is applied:
                - "before": PCA is applied before clustering.
                - "after": PCA is applied after clustering. Defaults to "before".

        Notes:
            - If `pca_timing` is "before", the KNN clustering is computed on PCA-reduced data.
            - If `pca_timing` is "after", the clustering is performed first, and then PCA reduces the data to 2 dimensions.
            - The PCA-reduced dimensions are labeled as `PCA_D1` and `PCA_D2`.
        """
        self.cluster_count = cluster_count
        if pca_timing == "before":
            self.data["cluster"] = self.calculate_knn(cluster_count, pca=True)
        else:
            data_pca = PCA(n_components=2).fit_transform(self.data)
            self.data["PCA_D1"] = data_pca[:, 0]
            self.data["PCA_D2"] = data_pca[:, 1]

        cluster_centers = (
            self.data[["PCA_D1", "PCA_D2", "cluster"]].groupby("cluster").mean()
        )

        groups = self.data[["PCA_D1", "PCA_D2", "cluster"]].groupby("cluster")

        fig = base_fig(title="KNN Clusters")

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
        colors = generate_colors(cluster_count)
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

        fig.show()

    def boxplot_of_cluster(self, cluster_number: int):

        if self.cluster_count is None:
            raise RuntimeError("Calculate KNN clusters first.")

        if cluster_number >= self.cluster_count:
            raise IndexError(
                f"You are trying to access the cluster with the index: {cluster_number}. But there are there are only {self.cluster_count} clusters. [starting with Cluster 0]"
            )

        fig = base_fig(title=f"Boxplot of Cluster {cluster_number}")

        pca_data = PCA(n_components=1).fit_transform(self.original_data)
        pca_data = pd.DataFrame(pca_data, columns=["PCA"]).reset_index(drop=True)
        pca_data["cluster"] = self.data["cluster"].reset_index(drop=True)
        pca_data["participant"] = self.data.index.tolist()

        cluster = pca_data[(pca_data["cluster"] == cluster_number)]
        colors = generate_colors(self.cluster_count)

        fig.add_trace(
            go.Box(
                y=cluster["PCA"],
                boxpoints="all",
                text=cluster["participant"],
                hovertemplate="Participant: %{text} <extra></extra>",
                name=f"Cluster {cluster_number}",
                marker=dict(color=colors[cluster_number]),
            )
        )

        fig.show()

    def boxplot_all_clusters(self):

        if self.cluster_count is None:
            raise RuntimeError("Calculate KNN clusters first.")

        fig = base_fig(title="Boxplot of all clusters", x_label="Cluster Number")

        pca_data = PCA(n_components=1).fit_transform(self.original_data)

        pca_data = pd.DataFrame(pca_data, columns=["PCA"]).reset_index(drop=True)
        pca_data["cluster"] = self.data["cluster"].reset_index(drop=True)
        pca_data["participant"] = self.data.index.tolist()

        groups = pca_data.groupby("cluster")
        colors = generate_colors(self.cluster_count)
        for idx, (group_name, df) in enumerate(groups):
            fig.add_trace(
                go.Box(
                    y=df["PCA"],
                    name=f"Cluster {group_name}",
                    text=df["participant"],
                    boxpoints="all",
                    hovertemplate="Participant: %{text} <extra></extra>",
                    marker=dict(color=colors[idx]),
                )
            )

        fig.show()

    def boxplot_clustered_image_perception(self):

        fig = base_fig(
            title="Boxplot of clustered images",
            x_label="Cluster Number",
            y_label="Mean Participants Rating",
        )

        # Gruppieren der Daten nach Clustern
        grouped_data = self.data.groupby("cluster")
        colors = generate_colors(self.cluster_count)
        for idx, (group_name, df) in enumerate(grouped_data):
            # Entfernung der Spalten PCA und Cluster
            df.drop(columns=["PCA_D1", "PCA_D2", "cluster"], inplace=True)
            df = df.mean()

            fig.add_trace(
                go.Box(
                    y=df,
                    name=f"Cluster {idx}",
                    boxpoints="all",
                    text=df.index.to_list(),
                    hovertemplate="Picture: %{text} <extra></extra>",
                    marker=dict(color=colors[idx]),
                )
            )

        fig.show()

    def find_outliers(self):
        grouped_data = self.data.groupby("cluster")

        for idx, (group_name, df) in enumerate(grouped_data):
            df.drop(columns=["PCA_D1", "PCA_D2", "cluster"], inplace=True)

            image_df = pd.DataFrame(data=df.mean(), columns=["means"])
            image_df["cluster_mean"] = df.to_numpy().mean()
            image_df["cluster_std"] = df.to_numpy().std()

            image_df["1xstd_outliers"] = (
                image_df["means"] > (image_df["cluster_mean"] + image_df["cluster_std"])
            ) | (
                image_df["means"] < (image_df["cluster_mean"] - image_df["cluster_std"])
            )

            self.image_analysis.append(image_df)

        self.show_pictures()

    def show_pictures(self):
        base_path = os.path.join(os.getcwd(), "data/images")

        image_set = []

        for idx in range(len(self.image_analysis)):
            group_outliers = self.image_analysis[idx]

            filtered_outliers = group_outliers[group_outliers["1xstd_outliers"] == True]
            image_numbers = filtered_outliers.index.tolist()
            images = [
                {
                    "data": Image.open(
                        os.path.join(base_path, f"{image:05}.png")
                    ).convert("RGB"),
                    "name": f"{image:05}.png",
                    "path": os.path.join(base_path, f"{image:05}.png"),
                    "mean": filtered_outliers.iloc[idx]["means"],
                }
                for idx, image in enumerate(image_numbers)
            ]

            image_set.append(images)

        number_of_rows = self.cluster_count
        max_number_of_columns = max(len(subarray) for subarray in image_set)
        subplot_counter = 1

        fig = plt.figure()
        for idx, images_of_one_cluster in enumerate(image_set):
            for image in images_of_one_cluster:
                plt.subplot(
                    number_of_rows,
                    max_number_of_columns,
                    subplot_counter,
                )
                plt.imshow(image["data"])
                plt.axis("off")
                # Text unten (z. B. Bildname)
                plt.text(
                    0.5,
                    -0.2,
                    image["name"],
                    ha="center",
                    fontsize=8,
                    transform=plt.gca().transAxes,
                )

                # Text oben (z. B. Zusatzinformation)
                plt.text(
                    0.5,
                    1.05,
                    # TODO: herausfinden ob negativ = stressig ist oder anders herum.
                    "positiv" if image["mean"] > 0 else "negativ",
                    color="red" if image["mean"] < 0 else "green",
                    ha="center",
                    fontsize=8,
                    transform=plt.gca().transAxes,
                )

                subplot_counter += 1
            index_offset_to_new_row = (
                max_number_of_columns * (idx + 1) - subplot_counter + 1
            )
            fig.text(
                0.5,
                0.9 - idx * 0.28,
                f"Cluster {idx}",
                ha="center",
                fontsize=11,
                weight="bold",
            )
            subplot_counter += index_offset_to_new_row

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
