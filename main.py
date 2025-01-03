import packages.image_analyse.feature_extraction as fe
from packages.visualization.plotly import histogram

path_for_images = "./data/Unruhig_Format16-9"

pictures = fe.get_images_and_convert_to_grayscale(path_for_images)

print(len(pictures))




