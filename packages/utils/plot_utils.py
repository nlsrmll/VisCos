import numpy as np
from matplotlib import pyplot as plt


def plot_images(image_list):
    num_images = len(image_list)
    max_per_plot = 4  # 2x2 Subplots pro Plot

    for i in range(0, num_images, max_per_plot):
        images_to_plot = image_list[
            i : i + max_per_plot
        ]  # Nimm max. 4 Bilder für diesen Plot
        num_images_in_this_plot = len(images_to_plot)

        # Bestimme Layout (1x1, 1x2 oder 2x2)
        if num_images_in_this_plot == 1:
            rows, cols = 1, 1
        elif num_images_in_this_plot == 2:
            rows, cols = 1, 2
        else:
            rows, cols = 2, 2

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 3, rows * 3)
        )  # Größe anpassen
        axes = np.array(axes).reshape(-1)  # Falls nötig, flach machen

        for ax, img in zip(axes, images_to_plot):
            ax.imshow(img)
            ax.axis("off")

        # Falls leere Subplots da sind, entfernen
        for ax in axes[len(images_to_plot) :]:
            fig.delaxes(ax)

        plt.show()
