from typing import List, Tuple

from sklearn.cluster import KMeans


# TODO: Assign value to data
def find_clusters_with_elbow(
    data, max_cluster_count: int
) -> Tuple[List[int], List[float]]:
    means = []
    inertias = []

    for cluster_count in range(1, max_cluster_count):
        # TODO: Control over random seed for getting same results every time
        k_means = KMeans(n_clusters=cluster_count, random_state=0)
        k_means.fit(data)

        means.append(cluster_count)
        inertias.append(k_means.inertia_)

    return means, inertias
