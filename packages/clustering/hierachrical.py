import numpy as np
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import ClusterNode, linkage


def hierarchical(data: np.ndarray):
    linkage_matrix = linkage(data, method="ward")

    fig = ff.create_dendrogram(linkage_matrix, orientation="bottom")

    print(fig["data"])

    fig.update_layout(title="Hierarchical Clustering")
    # fig.show()


def get_cluster_members(tree: ClusterNode, node_id, original_points):
    if node_id < original_points:
        return [node_id]
    else:
        left = get_cluster_members(tree, tree.left.id, original_points)
        right = get_cluster_members(tree, tree.right.id, original_points)
    return left + right
