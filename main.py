
from packages.image_analyse.DataReader import ImageDataFrame
from packages.image_analyse.feature_extraction import get_intensity, get_sharpness

path_for_images = "./data/Unruhig_Format16-9"

def hello_world(test):
    print("Hello World!")

images = ImageDataFrame(path_for_images)

images.apply_method_and_save_to_column(get_intensity)
images.apply_method_and_save_to_column(get_sharpness)



