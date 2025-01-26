from packages.clustering.knn import ClusterAnalyze
from packages.visualization.plotly import base_fig


class ClusterVisualizer:
    def __init__(self, cluster: ClusterAnalyze):
        self.cluster = cluster
        pass

    def show_clusters(self):

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
        pass
