import numpy as np
from matplotlib import pyplot as plt


def plot_images(image_list: list[dir], titel: str = "Default Titel"):
    num_images = len(image_list)
    max_per_plot = 4  # 2x2 Subplots pro Plot
    counter = 1
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

        fig, axes = plt.subplots(rows, cols, figsize=(5, 5))  # Größe anpassen
        axes = np.array(axes).reshape(-1)  # Falls nötig, flach machen
        fig.suptitle(titel)

        for ax, img in zip(axes, images_to_plot):
            ax.imshow(img["data"])
            ax.axis("off")

            ax.set_title(f"{counter}.")
            ax.text(
                0.5,
                -0.15,
                f"{img['image_name']} | {img['component_value']:.4f}",
                fontsize=8,
                va="center",
                ha="center",
                transform=ax.transAxes,
            )
            counter += 1

        # Falls leere Subplots da sind, entfernen
        for ax in axes[len(images_to_plot) :]:
            fig.delaxes(ax)

        plt.show()
