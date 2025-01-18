from packages.clustering.knn import ClusterAnalyze
from packages.image_analyse.DataReader import CSVDataFrame

path = "./data/likert_results.csv"

likert_data = CSVDataFrame(path, True)

cluster_analyzer = ClusterAnalyze(likert_data)
cluster_analyzer.seed = 42

cluster_analyzer.calculate_knn(cluster_count=3)

cluster_analyzer.find_outliers()
