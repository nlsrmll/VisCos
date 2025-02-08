import colorsys
from typing import List


def generate_color_palett(color_count: int) -> List[str]:
    """
    Generates a list of distinct colors in RGB format, evenly distributed across the HSV color space.

    Parameters:
        color_count (int): The number of distinct colors to generate.

    Returns:
        list: A list of color strings in the format "rgb(r, g, b)", where `r`, `g`, and `b` are values between 0 and 255.

    Notes:
        - Colors are generated using the HSV color model and then converted to RGB.
        - The `hue` value is evenly distributed across the range [0, 1] for `color_count` distinct colors.
    """
    colors = []
    for i in range(color_count):
        hue = i / color_count
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 0.85)
        colors.append(f"rgb({rgb[0]*255}, {rgb[1]*255}, {rgb[2]*255})")
    return colors
