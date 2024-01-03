import glob
import pickle

import numpy as np
import pandas as pd


def __add_columns_with_labels(df, label):
    return np.c_[df, label * np.ones(len(df))]


def load_data(
        path: str, num_samples_per_class: int
) -> (dict[int, str], pd.DataFrame, pd.DataFrame):
    labels = {}
    bitmaps_x = []
    bitmaps_y = []

    for index, file_name in enumerate(glob.glob(f"{path}/*.npy")):
        drawing_name = file_name.split("\\")[-1].split(".")[0]
        labels[index] = drawing_name

        bitmap = np.load(file_name)
        bitmap_labeled = __add_columns_with_labels(bitmap, index)

        bitmaps_x.append(bitmap_labeled[:num_samples_per_class, :-1])
        bitmaps_y.append(bitmap_labeled[:num_samples_per_class, -1])

        # __plot_samples(bitmap_labeled, 5, 5, title=drawing_name)

    # merge the arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
    x = np.concatenate(bitmaps_x, axis=0).astype("float32")  # all columns but the last
    y = np.concatenate(bitmaps_y, axis=0).astype("float32")  # the last column

    return labels, x, y


def save_model(model, model_name):
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
