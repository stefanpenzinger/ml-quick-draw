import numpy as np
from matplotlib import pyplot as plt


def plot_bitmap(bitmap_array):
    reshaped_array = bitmap_array.reshape((28, 28))
    plt.imshow(reshaped_array, cmap="gray")
    plt.title("Resized Canvas")
    plt.show()


plot_bitmap(np.load("drawings/bitmap-picture.npy"))
