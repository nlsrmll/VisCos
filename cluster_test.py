from models.file import File
from packages.image_analyse.DataReader import CSVDataFrame
from packages.study_data.analyzer import StudyAnalyzer
from packages.study_data.visualizer import StudyDataVisualizer


path = "./data/likert_results.csv"

likert_data = CSVDataFrame(path, True)


analyzer = StudyAnalyzer(likert_data)
study_visualizer = StudyDataVisualizer(analyzer)

analyzer.seed = 42
analyzer.visualize_hierarchical_dendrogram()
analyzer.calculate_hierarchical_clustering(cluster_count=2)

study_visualizer.cluster_as_scatter()


# study_visualizer.cluster_as_scatter()

# outliers = analyzer.find_outliers()
#
# outlier_list = [group.index.tolist() for group in outliers]
#
# singleList = [f"{element:05}" for singleList in outlier_list for element in singleList]
#
# image_outliers = ImageReader("./data/images", singleList)
# image_outliers.apply_method_and_save_to_column(get_contrast)
# image_outliers.apply_method_and_save_to_column(get_sharpness)
# image_outliers.apply_method_and_save_to_column(get_slope)
# image_outliers.apply_method_and_save_to_column(get_mean_brightness)
#
#
# excluded_picture = image_outliers.get_excluded_pictures()
# excluded_picture.apply_method_and_save_to_column(get_contrast)
# excluded_picture.apply_method_and_save_to_column(get_sharpness)
# excluded_picture.apply_method_and_save_to_column(get_slope)
# excluded_picture.apply_method_and_save_to_column(get_mean_brightness)
#
#
# fig = boxplot(dfs=[image_outliers, excluded_picture], column="mean_brightness")
#
# fig.show()
