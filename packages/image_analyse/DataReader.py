from enum import Enum
from os import PathLike
from typing import Tuple, List, Callable
import numpy as np
import os
import re

import pandas
from PIL import Image, ImageOps
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from packages.visualization.plotly import histogram_as_bar

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

#TODO: Über settings Anzahl der Nachkommastellen setzten

#TODO: Options die Kovertierung in grayscale regelt
class ImageReaderColumns(Enum):
    DATA = "data"

class ImageDataFrame:
    def __init__(self, image_path: PathLike[str] | str):
        images, image_names = get_images_and_convert_to_grayscale(image_path)
        self.data = pd.DataFrame({
            "picture_names": image_names,
            ImageReaderColumns.DATA.value: images
        })

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> DataFrame:
        for index in self.data.index:
            yield self.data.loc[index]

    def __getitem__(self, index:int) -> DataFrame:
        return self.data.loc[index]

    def get_data_frame_obj(self) -> pandas.DataFrame:
        return self.data

    def apply_method_and_save_to_column(self, method: Callable, column_name: str = None):

        if column_name is None:
            match = re.search(r"get_(\w+)", method.__name__)

            if match:
                column_name = match.group(1)

            else:
                raise NameError(f"Method name: {method.__name__} does not start with get_. Please rename the method or hand over a column name.")


        self.data[column_name] = self.data[ImageReaderColumns.DATA.value].apply(method)

    def show_histogram_of(self, column_name: str):

        try:
            data = self.data[column_name]
            histogram_as_bar(data=data, title=f"Histogram - {column_name.title()}", x_label=column_name.title(), y_label="Number of Images")
        except KeyError:
            raise KeyError(f"Column name: {column_name} does not exist.") from None




def get_images_and_convert_to_grayscale(path:str | PathLike[str]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads all images from a given folder, converts them to grayscale, and returns them as NumPy arrays.

    Parameters:
        path (str | PathLike[str]): Path to the folder containing the images.

    Returns:
        Tuple[List[np.ndarray], List[str]]:
            - A list of images converted to single-channel grayscale, represented as NumPy arrays.
            - A list of corresponding image file names.

    Notes:
        - Prints information about the folder, number of images, and the dimensions of the first image.
    """
    image_names = [f for f in os.listdir(path) if
                   os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]  # comparison case insensitive

    # read images one-by-one and convert to single channel grayscale
    images = [ImageOps.grayscale(Image.open(os.path.join(path, i))).convert('L') for i in image_names]

    # Print Info
    print('Folder Name: ' + os.path.split(path)[-1])
    print(f'|__ Number of images: {len(image_names)}')
    print(f'|__ Image dimension: {np.array(images[0]).shape}')
    print()

    return [np.array(ImageOps.grayscale(i)) for i in images], image_names
