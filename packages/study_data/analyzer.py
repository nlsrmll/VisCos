import os
import random
from typing import List

import pandas
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

from packages.utils.colors import generate_color_palett
from packages.visualization.plotly import base_fig, line_chart, scatter, dendrogram


class StudyAnalyzer:
    pca: PCA = None

    def __init__(self, data: pd.DataFrame, seed: int = None, cluster_count: int = None):
        self.data = data
        # TODO: Kann man hier nicht einfach die Columnspalten speichern,
        # sodass man nur diese aus dem DF herausnimmt? Spart speicher
        self.__original_columns = self.data.columns.tolist()
        self.seed = seed
        self.cluster_count = cluster_count
        self.image_analysis: List[pandas.DataFrame] = []

    def update_cache(self):
        self.__original_columns = self.data.columns.tolist()

    def visualize_elbow_curve(self, cluster_count: int) -> None:

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

        line_chart(
            means,
            inertias,
            title="Elbow Curve for K-Nearest Neighbors",
            x_label="Cluster",
            y_label="Inertia",
        )

    def visualize_silhouette_score(self, cluster_count: int) -> None:

        scores = []

        for k in range(2, cluster_count + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=(
                    self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
                ),
            )
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            scores.append(score)

        line_chart(
            x=[i for i in range(2, cluster_count + 1)],
            y=scores,
            title="Silhouette Score for KMeans",
            x_label="Number of Clusters",
            y_label="Silhouette Score",
        )

    def visualize_hierarchical_dendrogram(self):
        plt.figure()
        linkage = sch.linkage(self.data[self.__original_columns], method="ward")
        sch.dendrogram(linkage)
        plt.show()

    def calculate_pca(self, n_components: int, **kwargs):

        if self.pca is None:
            self.pca = PCA(n_components=n_components, **kwargs)

            scores = self.pca.fit_transform(self.data[self.__original_columns])
            for idx in range(n_components):
                self.data[f"PCA_D{idx + 1}"] = scores[:, idx]
        else:
            self.data.drop(
                columns=[f"PCA_D{idx + 1}" for idx in range(self.pca.n_components_)]
            )

            self.pca = PCA(n_components=n_components, **kwargs)

            scores = self.pca.fit_transform(self.data[self.__original_columns])
            for idx in range(n_components):
                self.data[f"PCA_D{idx + 1}"] = scores[:, idx]

    def calculate_t_SNE(self, n_components: int, **kwargs):
        t_sne_result = TSNE(n_components=n_components, **kwargs).fit_transform(
            self.data[self.__original_columns]
        )

        for idx in range(n_components):
            self.data[f"t_SNE_D{idx + 1}"] = t_sne_result[:, idx]

    def calculate_kmeans(
        self,
        cluster_count: int,
    ):
        self.cluster_count = cluster_count
        k_means = KMeans(
            n_clusters=cluster_count,
            random_state=(
                self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
            ),
        )

        if self.pca is not None:
            labels = k_means.fit_predict(
                self.data[[f"PCA_D{idx+1}" for idx in range(self.pca.n_components)]]
            )
        else:
            labels = k_means.fit_predict(self.data)

        self.data["cluster"] = labels

    def calculate_hierarchical_clustering(self, cluster_count: int):
        hc_labels = sch.fcluster(
            sch.linkage(self.data[self.__original_columns], method="ward"),
            self.cluster_count,
            criterion="maxclust",
        )
        self.data["hc_labels"] = hc_labels

    def boxplot_clustered_image_perception(self):

        fig = base_fig(
            title="Boxplot of clustered images",
            x_label="Cluster Number",
            y_label="Mean Participants Rating",
        )

        # Gruppieren der Daten nach Clustern
        grouped_data = self.data.groupby("cluster")
        colors = generate_color_palett(self.cluster_count)
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

    # TODO: Visualisierung rausnehmen
    def find_outliers(self, shouldShow: bool = False):
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

        if shouldShow:
            self.show_pictures()

        return [
            cluster[cluster["1xstd_outliers"] == True]
            for cluster in self.image_analysis
        ]

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

        fig = plt.figure(facecolor="white")

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
                    -0.1,
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

        # Globaler Text unten
        fig.text(
            0.5,
            0.01,
            "Upper and lower border equals σ±x̄",
            ha="center",
            fontsize=8,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
