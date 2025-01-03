import packages.image_analyse.feature_extraction as fe
import packages.visualization.plotly as plt

path_for_images = "./data/Unruhig_Format16-9"

pictures, picture_names = fe.get_images_and_convert_to_grayscale(path_for_images)

data = []

for picture in pictures:
    data.append(fe.get_contrast(picture))


plt.histogram_as_bar(data)


