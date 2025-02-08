from __future__ import annotations

import os
import re
from os import PathLike
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from packages.utils import AvailableImages_Dir

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class ImageReader(pd.DataFrame):
    applied_methods = []

    def __init__(self, path: PathLike[str] | str, image_list: list[str] = None):
        if image_list is None:
            images, image_names = get_images_and_convert_to_grayscale(path)
        else:
            images, image_names = get_images_and_convert_to_grayscale(path, image_list)

        super().__init__(pd.DataFrame({"picture_names": image_names, "data": images}))
        self.path = path
        self.folder_name = self.path.split("/")[-1]

    def get_excluded_pictures(self) -> ImageReader:

        outer_quantity = list(
            set(get_file_names_from_dir(self.path)) - set(self["picture_names"])
        )
        return ImageReader(path=self.path, image_list=outer_quantity)

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

        self[column_name] = self["data"].apply(method)
        self.applied_methods.append(column_name)

    def norm_values(self, method: Literal["minmax", "zscore"] = "minmax"):
        scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        self[self.applied_methods] = scaler.fit_transform(self[self.applied_methods])

    def get_features(self):
        return self[self.applied_methods]


def remove_extension(filename: str) -> str:
    for ext in IMAGE_EXTENSIONS:
        if filename.endswith(ext):
            return filename.removesuffix(ext)
    return filename  # Falls keine Endung vorhanden ist, bleibt der Name gleich


def get_images_and_convert_to_grayscale(
    path: Union[str, os.PathLike],
    image_list: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads images from a specified folder, converts them to grayscale, and returns them as NumPy arrays.

    Parameters:
        path (Union[str, os.PathLike]): The path to the folder containing the images.
        image_list (Optional[List[str]], optional): A list of image file names (without extensions) to load.
                                                   If provided, only these images will be processed. Defaults to None.

    Returns:
        Tuple[List[np.ndarray], List[str]]:
            - A list of images converted to grayscale as NumPy arrays.
            - A list of corresponding image file names.

    Notes:
        - Only images with extensions listed in `IMAGE_EXTENSIONS` are processed.
        - The function ensures that the provided `path` is a `Path` object.
    """
    path = Path(path)  # Sicherstellen, dass path ein Path-Objekt ist

    # Alle Bilddateien mit den erlaubten Endungen abrufen
    image_names = [
        f.name for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Falls image_list angegeben ist: Filtere nur relevante Bilder
    if image_list is not None:
        cleaned_image_list = {remove_extension(name) for name in image_list}
        image_names = [
            name for name in image_names if remove_extension(name) in cleaned_image_list
        ]

    # Lade Bilder als Graustufen
    images = [
        np.array(ImageOps.grayscale(Image.open(path / img))) for img in image_names
    ]

    # Print Info
    print(f"Folder Name: {path.name}")
    print(f"|__ Number of images: {len(image_names)}")
    print(f"|__ Image dimension: {images[0].shape if images else 'No images found'}")
    print()

    return images, image_names


def get_file_names_from_dir(path) -> List[str]:
    image_names = [
        f
        for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]  # comparison case insensitive
    return image_names


def get_full_image_name(image_name: str | int | numpy.int_) -> str:
    if isinstance(image_name, numpy.int_) or isinstance(image_name, int):
        image_name = str(image_name)

    if len(image_name) != 5 and not image_name.endswith(".*"):
        image_name = image_name.zfill(5)

    try:
        return AvailableImages_Dir[image_name]
    except KeyError:
        raise KeyError(f"Could not find the image {image_name}")
