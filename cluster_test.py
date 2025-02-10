from packages.image_analyse.DataReader import CSVDataFrame
from packages.study_data.analyzer import StudyAnalyzer
from packages.study_data.visualizer import StudyDataVisualizer


path = "./data/likert_results.csv"

likert_data = CSVDataFrame(path, True)


analyzer = StudyAnalyzer(likert_data)
study_visualizer = StudyDataVisualizer(analyzer)

analyzer.seed = 42
analyzer.calculate_pca(2)
analyzer.calculate_kmeans(3)

study_visualizer.show_cluster_votings()
