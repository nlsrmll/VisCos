
from packages.image_analyse.DataReader import ImageDataFrame
from packages.image_analyse.feature_extraction import get_contrast

path_for_images = "./data/Unruhig_Format16-9"


images = ImageDataFrame(path_for_images)


images.apply_method_and_save_to_column(get_contrast)



images.show_histogram_of("contrast")