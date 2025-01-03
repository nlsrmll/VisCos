import os
from scipy import signal
from os import PathLike
from typing import List, Tuple, Literal

from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
from numpy.fft import fftshift, fft2
from skimage.feature import local_binary_pattern

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

#TODO: Über settings Anzahl der Nachkommastellen setzten

#TODO: Options die Kovertierung in grayscale regelt
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
    # image_names = [f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSION.lower())]  # comparison case insensitive
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

def get_intensity(image: np.ndarray) -> np.floating:
    """
    Calculates the average intensity of an image.

    Parameters:
        image (np.ndarray): A NumPy array representing the image

    Returns:
        np.floating: The mean intensity of the image.
    """
    return np.mean(image)

def get_sharpness(image: np.ndarray) -> float:
    """
    Computes the sharpness of an image using the variance of the Laplacian.

    Parameters:
        image (np.ndarray): A NumPy array representing the image.

    Returns:
        float: The sharpness of the image, measured as the variance of the Laplacian.
    """
    return cv.Laplacian(image, cv.CV_32F).var()

def get_contrast(image: np.ndarray) -> np.floating:
    """
    Calculates the contrast of an image using the standard deviation of pixel intensities.

    Parameters:
        image (np.ndarray): A NumPy array representing the image.

    Returns:
        np.floating: The contrast of the image, measured as the standard deviation of pixel values.
    """
    return np.std(image)

def get_bw_ration(image: np.ndarray) -> np.floating:
    """
    Calculates the black-to-white pixel ratio in a binary or grayscale image.

    Parameters:
        image (np.ndarray): A NumPy array representing the image, where pixel values range from 0 to 255.

    Returns:
        np.floating: The ratio of black pixels (pixel values < 128) to white pixels (pixel values >= 128).
    """
    black_pixels = np.sum(image < 128)
    white_pixels = np.sum(image >= 128)
    return black_pixels / white_pixels

def get_bw_ratio_otsu(image: np.ndarray) -> np.floating:
    """
    Calculates the black-to-white pixel ratio in an image using Otsu's thresholding method.

    Parameters:
        image (np.ndarray): A NumPy array representing the image, where pixel values range from 0 to 255.

    Returns:
        np.floating: The ratio of black pixels (pixel values below the Otsu threshold)
                     to white pixels (pixel values equal to or above the Otsu threshold).
    """
    thresh, _ = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

    black_pixels = np.sum(image < thresh)
    white_pixels = np.sum(image >= thresh)

    return black_pixels / white_pixels

def get_entropy(image: np.ndarray) -> float:
    """
    Calculates the entropy of an image, a measure of its information content.

    Parameters:
        image (np.ndarray): A NumPy array representing the image, where pixel values range from 0 to 255.

    Returns:
        float: The entropy of the image, calculated based on the normalized histogram of pixel intensities.
    """

    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 256), density=True)
    entropy = 0
    for p in histogram:
        if p != 0:
            entropy -= p * np.log2(p)

    return entropy

def get_histogram(image: np.ndarray, bins_:int, range_:Tuple[int, int]=(0, 255)) -> np.ndarray:
    """
    Computes the histogram of an image.

    Parameters:
        image (np.ndarray): A NumPy array representing the image.
        bins_ (int): The number of bins for the histogram.
        range_ (Tuple[int, int], optional): The range of pixel intensity values to consider. Defaults to (0, 255).

    Returns:
        np.ndarray: An array containing the histogram values for each bin.
    """
    return np.histogram(image.ravel(), bins=bins_, range=range_, density=False)[0]


def get_image_spectrum(image: np.ndarray, nbins:int=100, lowcut:int=2):
    """
    Computes the amplitude spectrum of an image and fits a line to the log-log plot
    of amplitude vs. frequency. This function provides information about the frequency
    content of the image.

    Parameters:
        image (np.ndarray): A 2D array representing the grayscale image.
        nbins (int, optional): Number of bins for quantizing frequency. Defaults to 100.
        lowcut (int, optional): Percentage cutoff for low frequencies in the fit. Defaults to 2.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
            - amp (np.ndarray): Amplitude spectrum quantized into bins.
            - frequency (np.ndarray): Frequency scale for each quantized frequency bin.
            - fst (int): Index of the first bin beyond the specified lowcut percentage.
            - p (np.ndarray): Coefficients of the fitted line to the log-log plot of amplitude vs. frequency.

    Notes:
        - Original Code by Peter Kovesi (www.peterkovesi.com).
        - See the `imspec()` function for more details.
    """
    amp = np.zeros(nbins)
    fcount = np.ones(nbins)

    # Generate a matrix 'radius' every element of which has a value
    # given by its distance from the centre.  This is used to index
    # the frequency values in the spectrum.
    mag = fftshift(np.abs(fft2(image)))
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # The following fiddles the origin to the correct position
    # depending on whether we have and even or odd size.
    # In addition, the values of x and y are normalised to +- 0.5

    if cols % 2 == 0:
        x = (x - cols / 2 - 1) / cols
    else:
        x = (x - (cols + 1) / 2) / cols
    if rows % 2 == 0:
        y = (y - rows / 2 - 1) / rows
    else:
        y = (y - (rows + 1) / 2) / rows

    radius = np.sqrt(x ** 2 + y ** 2)
    radius = np.round(radius / np.max(radius) * (nbins - 1))

    radius = radius.astype(np.int64)
    for r in range(rows):
        for c in range(cols):
            ind = radius[r, c]
            amp[ind] += mag[r, c]
            fcount[ind] += 1

    eps = 1e-12
    amp = amp / fcount + eps

    # Generate corrected frequency scale for each quantised frequency bin.
    # Note that the maximum frequency is sqrt(0.5) corresponding to the
    # points in the corners of the spectrum
    frequency = np.arange(nbins) / (nbins - 1) * (np.sqrt(.5))

    # Find first index value beyond the specified histogram cutoff
    fst = int(round(nbins * lowcut / 100 + 1))
    # Get Line fitted to the amplitude and frequency
    p = np.polyfit(np.log(frequency[fst:]), np.log(amp[fst:]), 1)

    return amp, frequency, fst, p

def get_slope (image:np.ndarray) -> float:
    """
    Calculates the slope of the log-log plot of amplitude vs. frequency from the image spectrum.

    Steps:
    1. Resizes the input image to a fixed size using k-nearest neighbor interpolation.
    2. Applies a Hanning window to reduce edge artifacts in the frequency domain.
    3. Computes the image spectrum and fits a line to the log-log plot of amplitude vs. frequency.

    Parameters:
        image (np.ndarray): A 2D array representing the grayscale image.

    Returns:
        float: The slope of the fitted line from the log-log plot of amplitude vs. frequency.
    """
    resized_image = resize_image_with_k_nearst(image)
    hann_image = apply_hanning_window(resized_image)
    _, _, _, p = get_image_spectrum(hann_image)
    return p[0]

def resize_image_with_k_nearst(image: np.ndarray, size:Tuple[int, int]=(256, 256)) -> np.ndarray:
    """
    Resizes an image to the specified size using k-nearest neighbor interpolation.

    Parameters:
        image (np.ndarray): A NumPy array representing the image to be resized.
        size (Tuple[int, int], optional): The target size of the image as (width, height). Defaults to (256, 256).

    Returns:
        np.ndarray: The resized image as a NumPy array.
    """
    pil_imag = Image.fromarray(image)
    resized_image = pil_imag.resize(size, resample=Image.NEAREST)
    return np.asarray(resized_image)

def apply_hanning_window(image: np.ndarray) -> np.ndarray:
    """
    Applies a 2D Hanning window to an image to reduce edge artifacts in the frequency domain.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the image.

    Returns:
        np.ndarray: The image with the 2D Hanning window applied.
    """
    hann_window = signal.windows.hann(image.shape[0], sym=False)
    hann_window2d = np.outer(hann_window, hann_window)
    return image * hann_window2d

def get_lbp_histogram(image:np.ndarray, P_:int=24, R_:int=3) -> np.ndarray:
    """
    Computes the Local Binary Pattern (LBP) histogram of an image.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.
        P_ (int, optional): Number of circularly symmetric neighbor points to consider. Defaults to 24.
        R_ (int, optional): Radius of the circle. Defaults to 3.

    Returns:
        np.ndarray: The histogram of the LBP image.
    """
    lbp = local_binary_pattern(image, P_, R_, method='uniform')
    n_bins = int(lbp.max() + 1)
    return get_histogram(lbp, n_bins, (0, n_bins))


def get_fractal_dimension(image:np.ndarray, threshold:float) -> float:
    """
    Computes the fractal dimension of a 2D image using the box-counting method.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.
        threshold (float): Threshold value (between 0 and 1) to binarize the image.
                           Pixels below `threshold * maximum_pixel_value` are considered background.

    Returns:
        float: The estimated fractal dimension of the image.

    Notes:
        - The function assumes the input is a 2D image.
        - Based on the box-counting method for fractal dimension estimation.
        - Adapted from: https://stackoverflow.com/questions/44793221/python-fractal-box-count-fractal-dimension
    """
    assert (len(image.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(np_img, k):
        S = np.add.reduceat(
            np.add.reduceat(np_img, np.arange(0, np_img.shape[0], k), axis=0),
            np.arange(0, np_img.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform np_img into a binary array
    maximum = np.max(image)  # better: take theoretical maximum
    image = (image < threshold * maximum)

    # Minimal dimension of image
    p = min(image.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        h = boxcount(image, size)
        if h == 0:
            h = 1
        counts.append(h)

    # Fit the successive log(sizes) with log (counts)
    coefficients = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coefficients[0]


def get_edges(image:np.ndarray) -> np.ndarray:
    """
    Detects edges in an image using the Canny edge detection algorithm.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        np.ndarray: A binary image with detected edges, where edge pixels are set to 255 and non-edge pixels are set to 0.
    """
    edges = cv.Canny(image, threshold1=85, threshold2=255)
    return edges


def get_edge_density(image:np.ndarray) -> float:
    """
    Calculates the edge density of an image, which is the ratio of edge pixels to the total number of pixels.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        float: The edge density, defined as the number of edge pixels divided by the total number of pixels in the image.
    """
    edges = get_edges(image)

    return np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])


def get_corners(image:np.ndarray, block_size:int = 2, kernel_size:int=3, k:int=0.04) -> np.ndarray:
    """
    Detects corners in an image using the Harris corner detection algorithm.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.
        block_size (int, optional): The size of the neighborhood considered for corner detection. Defaults to 2.
        kernel_size (int, optional): Aperture parameter of the Sobel derivative used. Defaults to 3.
        k (float, optional): Harris detector free parameter, typically between 0.04 and 0.06. Defaults to 0.04.

    Returns:
        np.ndarray: A binary mask where corner pixels are set to 1, and non-corner pixels are set to 0.
    """

    corners = cv.cornerHarris(image, blockSize=block_size, ksize=kernel_size, k=k)


    corners = cv.dilate(corners, None)
    threshold = 0.01 * corners.max()
    corner_mask = corners > threshold
    corners = corner_mask.astype(np.uint8)

    return corners

def get_corner_density(image: np.ndarray) -> float:
    """
    Calculates the corner density of a grayscale image, which is the proportion of pixels that are detected as corners.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.
        b_Debug_Output (bool, optional): If True, displays debug information. Defaults to False.

    Returns:
        float: The corner density, defined as the number of corner pixels divided by the total number of pixels in the image.

    Notes:
        - Source: Projektgruppe Imagitation
    """
    corners = get_corners(image)

    return np.count_nonzero(corners) / (image.shape[0] * image.shape[1])


def get_corner_portion(image:np.ndarray) -> float:
    """
    Calculates the ratio of corner pixels to edge pixels in a grayscale image.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        float: The ratio of corner pixels to edge pixels. If there are no edge pixels, the ratio is 0.
    """
    edges = get_edges(image)
    corners = get_corners(image)

    cnt_edges = np.count_nonzero(edges)

    ratio = 0
    if 0 < cnt_edges:
        ratio = np.count_nonzero(corners) / cnt_edges

    return ratio


def compute_contours(image:np.ndarray):
    """
    Computes the contours of a binary image after resizing and preprocessing.

    Parameters:
        image (np.ndarray): A NumPy array representing the input grayscale image.

    Returns:
        list: A list of detected contours. Each contour is an array of points representing the boundary.
    """

    # Fixed image size for processing. sx_img_fixed:sy_img_fixed should be 16:9
    sx_img_fixed = 1280
    sy_img_fixed = 720

    img = cv.resize(image, (sx_img_fixed, sy_img_fixed), interpolation=cv.INTER_LINEAR)

    thresh, img_bin = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    # we ensure that there are fewer foreground (white) than background (black) pixels
    cnt_foreground_pixel = cv.countNonZero(img_bin)
    cnt_background_pixel = img_bin.size - cnt_foreground_pixel
    if cnt_background_pixel < cnt_foreground_pixel:
        img_bin = cv.bitwise_not(img_bin)

    # add padding to avoid problems in the computation (e.g., contours around the hole image)
    sz_border = 4
    # ruff: noqa
    img_bin_border = cv.copyMakeBorder(img_bin, sz_border, sz_border, sz_border, sz_border, cv.BORDER_CONSTANT,
                                      value=(0, 0, 0))
    # ruff: noqa


    contours, _ = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    return contours


def get_object_contours(image:np.ndarray):
    """
     Extracts object contours from a grayscale image. Only contours exceeding a minimum length
     are retained as object contours.

     Parameters:
         image (np.ndarray): A 2D NumPy array representing the grayscale image.

     Returns:
         list: A list of contours that exceed the minimum length threshold. Each contour is an array of points.

     """
    to_ret_object_contours = []

    contours = compute_contours(image)

    min_contour_length_for_object = 200  # Parameter, works in conjunction with the parameters in compute_contours, see above

    for contour in contours:
        perimeter = cv.arcLength(contour, closed=True)

        if perimeter > min_contour_length_for_object:
            to_ret_object_contours.append(contour)

    return to_ret_object_contours


def get_object_count(image:np.ndarray) -> int:
    """
    Calculates the number of objects in a grayscale image. Objects are defined as contours
    that exceed a minimum length threshold.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        int: The number of detected objects (contours exceeding the minimum length).
    """
    object_contours = get_object_contours(image)

    return len(object_contours)



def get_object_portion(image:np.ndarray) -> float:
    """
    Calculates the proportion of objects (large contours) to all detected contours in a grayscale image.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        float: The proportion of object contours (large contours) to all contours, as a value between 0 and 1.
               If no contours are detected, the proportion is 0.
    """
    contours = compute_contours(image)
    object_contours = get_object_contours(image)

    return len(object_contours) / len(contours)



def get_mean_object_similarity(image:np.ndarray) -> float:
    """
    Calculates the mean similarity between all detected objects (contours) in a grayscale image.
    The similarity is computed using Hu moments. Lower values correspond to higher similarity.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        float: The mean similarity between all objects. If no valid similarity can be calculated, returns -1.

    Notes:
        - Source: Projektgruppe Imagitation
    """
    object_contours = get_object_contours(image)

    similarity = []
    for i in range(len(object_contours)):
        for j in range(len(object_contours)):
            if i != j:
                sim = cv.matchShapes(object_contours[i], object_contours[j], cv.CONTOURS_MATCH_I1, 0)
                similarity.append(sim)

    # calculate average

    return (sum(similarity) / len(similarity)) if (
                (np.isinf(sum(similarity)) is False) and (len(similarity) > 0)) else -1



def compute_circularity(contours:list)-> list:
    """
    Computes the circularity of each contour in a given list of contours.
    Circularity is defined as: \( 4 \pi \times \text{Area} / \text{Perimeter}^2 \),
    where a value closer to 1 indicates a perfect circle.

    Parameters:
        contours (list): A list of contours, where each contour is an array of points.

    Returns:
        list: A list of circularity values for each contour, with values in the range [0, 1].

    Notes:
        - Source: Projektgruppe Imagitation
        - Circularity for contours with a perimeter of 0 is set to 0.0.
    """
    circularity = []
    for i in range(len(contours)):
        perim = cv.arcLength(contours[i], True)
        area = cv.contourArea(contours[i])
        if perim != 0:
            circ = 4 * np.pi * area / perim ** 2
            circularity.append(circ)
        else:
            circularity.append(0.0)

    return circularity



def get_mean_object_circularity(image:np.ndarray)-> np.floating:
    """
    Calculates the mean circularity of all detected objects in a grayscale image.
    Circularity is computed for each object and averaged over all objects.

    Parameters:
        image (np.ndarray): A 2D NumPy array representing the grayscale image.

    Returns:
        float: The mean circularity of all detected objects. Circularity values are in the range [0, 1],
               where 1 indicates a perfect circle. If no objects are detected, returns NaN.
    """
    object_contours = get_object_contours(image)

    circularity = compute_circularity(object_contours)

    return np.mean(circularity)


def detect_hand_crafted_shapes(contours:list, mode:Literal['countVertices', 'fitShapes']='fitShapes'):
    """
       Detects the geometric type of each contour, such as triangle, square, circle, etc.,
       based on handcrafted rules and the specified mode.

       Parameters:
           contours (list): A list of contours, where each contour is an array of points.
           mode (Literal['countVertices', 'fitShapes'], optional): The detection mode:
               - 'countVertices': Classifies shapes based on the number of vertices.
               - 'fitShapes': Fits geometric shapes and evaluates their properties.
             Defaults to 'fitShapes'.

       Returns:
           tuple:
               - labels (list): A list of integers representing the detected shape type for each contour.
               - unique_labels (tuple): A tuple of shape type names corresponding to the labels.

       Modes:
           - `countVertices`:
               - triangle: 3 vertices
               - quadrangle: 4 vertices
               - ellipse: Circular contours with high circularity and > 4 vertices
               - other: None of the above
           - `fitShapes`:
               - triangle: Closely matches a triangle's properties
               - square: Rectangle with nearly equal sides
               - rectangle: Non-square rectangle
               - circle: High circularity and symmetry
               - ellipse: Elliptical shape but not a circle
               - other: None of the above

       Notes:
           - The function uses circularity and fitting methods to classify contours.
           - Invalid modes will raise a ValueError.
       """

    if mode == 'countVertices':
        unique_labels = ('triangle', 'quadrangle', 'ellipse', 'other')
    else:
        unique_labels = ('triangle', 'square', 'rectangle', 'circle', 'ellipse', 'other')

    labels = []
    circularities = compute_circularity(contours)

    for i, contour in enumerate(contours):

        num_vertices = len(contour)

        if mode == 'countVertices':

            if num_vertices == 3:
                labels.append(0)
            elif num_vertices == 4:
                labels.append(1)
            elif num_vertices > 4 and circularities[i] > 0.9:
                labels.append(2)
            else:
                labels.append(3)

        if mode == 'fitShapes':

            # assume contour is 'other'
            label = 5

            if num_vertices <= 10:

                # calculate contour area
                contour_area = cv.contourArea(contour)

                # fit rectangle
                rect = cv.minAreaRect(contour)
                rect_area = rect[1][0] * rect[1][1]
                rect_diff = abs(1 - (rect_area / contour_area))

                # fit triangle
                _, triangle = cv.minEnclosingTriangle(contour)

                if triangle is not None:
                    triangle_area = cv.contourArea(triangle)
                    triangle_diff = abs(1 - (triangle_area / contour_area))

                if triangle is not None and triangle_diff < rect_diff and triangle_diff < 0.1:
                    # contour is a triangle
                    label = 0
                elif rect_diff < triangle_diff and rect_diff < 0.1:

                    # contour is a rectangle or square
                    label = 2

                    # calculate ratio of rectangle sides
                    rect_ratio = rect[1][0] / rect[1][1]

                    # check if rectangle is a square
                    if abs(1 - rect_ratio) < 0.2:
                        # contour is a square
                        label = 1

            elif circularities[i] > 0.8:

                # fitEllipse
                ellipse = cv.fitEllipse(contour)

                # ellipse area is pi * a * b
                a = ellipse[1][0] / 2
                b = ellipse[1][1] / 2
                ellipse_area = np.pi * a * b

                # calculate contour area
                contour_area = cv.contourArea(contour)

                # calculate ratio of ellipse area to contour area
                area_ratio = ellipse_area / contour_area

                # check if contour is an ellipse
                if abs(1 - area_ratio) < 0.1:
                    # contour is an ellipse
                    label = 4

                    # ellipse is a circle
                    axes_ratio = a / b
                    if abs(1 - axes_ratio) < 0.1:
                        # append circle label
                        label = 3

            labels.append(label)

    return labels, unique_labels