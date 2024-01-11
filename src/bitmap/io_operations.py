import glob
import pickle

import numpy as np

from plot import plot_samples
from constants import (
    CNN_KEY,
    MODEL_PATH,
    DATA_PATH,
)


def __add_columns_with_labels(df, label):
    return np.c_[df, label * np.ones(len(df))]


def load_data(num_samples_per_class: int, should_plot: bool = False):
    labels = {}
    bitmaps_x = []
    bitmaps_y = []

    for index, file_name in enumerate(glob.glob(f"{DATA_PATH}/*.npy")):
        drawing_name = file_name.split("\\")[-1].split(".")[0]
        labels[index] = drawing_name

        bitmap = np.load(file_name)
        bitmap_labeled = __add_columns_with_labels(bitmap, index)

        if should_plot:
            plot_samples(bitmap_labeled, 5, 5, title=drawing_name)

        bitmaps_x.append(bitmap_labeled[:num_samples_per_class, :-1])
        bitmaps_y.append(bitmap_labeled[:num_samples_per_class, -1])

    # merge the arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
    x = np.concatenate(bitmaps_x, axis=0).astype("float32")  # all columns but the last
    y = np.concatenate(bitmaps_y, axis=0).astype("float32")  # the last column

    return labels, x, y


def save_model(model, model_name):
    if model_name == CNN_KEY:
        model.save(f"{MODEL_PATH}/{model_name}")
    else:
        with open(f"{MODEL_PATH}/{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
