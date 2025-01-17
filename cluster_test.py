from packages.clustering.knn import ClusterAnalyze
from packages.image_analyse.DataReader import CSVDataFrame

path = "./data/likert_results.csv"

likert_data = CSVDataFrame(path, True)
likert_data.remove_row("912858", "30117")

cluster_analyzer = ClusterAnalyze(likert_data)
cluster_analyzer.seed = 42

cluster_analyzer.visualize_knn(cluster_count=3)

cluster_analyzer.boxplot_all_clusters()
