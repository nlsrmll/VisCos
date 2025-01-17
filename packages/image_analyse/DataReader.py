import os
import re
from enum import Enum
from os import PathLike
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from PIL import Image, ImageOps

from packages.visualization.plotly import histogram

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# TODO: Über settings Anzahl der Nachkommastellen setzten


# TODO: Options die Kovertierung in grayscale regelt
class ImageReaderColumns(Enum):
    DATA = "data"


class ImageDataFrame:

    def __init__(self, image_path: PathLike[str] | str):
        self.path = image_path
        self.folder_name = self.path.split("/")[-1]
        images, image_names = get_images_and_convert_to_grayscale(image_path)
        self.data = pd.DataFrame(
            {"picture_names": image_names, ImageReaderColumns.DATA.value: images}
        )

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> DataFrame:
        for index in self.data.index:
            yield self.data.loc[index]

    def __getitem__(self, index: int) -> DataFrame:
        return self.data.loc[index]

    def get_data_frame_obj(self) -> pd.DataFrame:
        """
        Returns the current dataset as a pandas DataFrame.

        Returns:
            DataFrame: The DataFrame object containing the dataset.

        Notes:
            - This function provides direct access to the `self.data` attribute, which stores the dataset.
        """
        return self.data

    def apply_method_and_save_to_column(
        self, method: Callable, column_name: str = None
    ):
        """
        Applies a given method to a column in the dataset and saves the results to a new column.

        Parameters:
            method (Callable): A function or method to apply to the data. The method must accept one argument.
            column_name (str, optional): The name of the column where the results will be stored.
                                         If not provided, the column name is inferred from the method name.
                                         Specifically, the method name must start with "get_" to allow inference.

        Raises:
            KeyError: If `column_name` is not provided and the method name does not start with "get_".

        Notes:
            - If `column_name` is None, the function attempts to infer the column name by extracting the part
              of the method name after "get_".
            - The specified or inferred column name is added to the dataset (`self.data`) with the computed results.
            - The method is applied to each element in the column `ImageReaderColumns.DATA.value`.
        """
        if column_name is None:
            match = re.search(r"get_(\w+)", method.__name__)

            if match:
                column_name = match.group(1)

            else:
                raise KeyError(
                    f"Method name: {method.__name__} does not start with get_. Please rename the method or hand over a column name."
                ) from None

        self.data[column_name] = self.data[ImageReaderColumns.DATA.value].apply(method)

    def show_histogram_of(self, column_name: str):
        """
        Displays a histogram for a specified column in the dataset.

        Parameters:
            column_name (str): The name of the column to visualize as a histogram.

        Raises:
            KeyError: If the specified column name does not exist in the dataset.

        Notes:
            - The function utilizes `histogram` to generate the visualization.
            - The column data must exist in `self.data` and should be numerical or categorical to create a meaningful histogram.
        """
        try:
            histogram(
                df=self.data[["picture_names", column_name]],
                title=f"Histogram - {column_name}",
                x_label=column_name,
                y_label="Number of Images",
            )
        except KeyError:
            raise KeyError(f"Column name: {column_name} does not exist.") from None


class CSVDataFrame:
    def __init__(self, dataPath: PathLike[str] | str):
        self.path = dataPath
        self.folder_name = self.path.split("/")[-1]
        self.data = pd.read_csv(dataPath)
        print(self.data.head())


def get_images_and_convert_to_grayscale(
    path: str | PathLike[str],
) -> Tuple[List[np.ndarray], List[str]]:
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
    image_names = [
        f
        for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]  # comparison case insensitive

    # read images one-by-one and convert to single channel grayscale
    images = [
        ImageOps.grayscale(Image.open(os.path.join(path, i))).convert("L")
        for i in image_names
    ]

    # Print Info
    print("Folder Name: " + os.path.split(path)[-1])
    print(f"|__ Number of images: {len(image_names)}")
    print(f"|__ Image dimension: {np.array(images[0]).shape}")
    print()

    return [np.array(ImageOps.grayscale(i)) for i in images], image_names
