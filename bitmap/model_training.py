import numpy as np
import pandas as pd

import itertools

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as keras_backend


def __add_columns_with_labels(df, label):
    return np.c_[df, label * np.ones(len(df))]


def __plot_samples(input_array, rows=4, cols=5, title=""):
    """
    Function to plot 28x28 pixel drawings that are stored in a numpy array.
    Specify how many rows and cols of pictures to display (default 4x5).
    If the array contains fewer images than subplots selected, surplus subplots remain empty.
    """

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.axis("off")
    plt.title(f"{title}\n")

    for i in list(range(0, min(len(input_array), (rows * cols)))):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(
            input_array[i, :784].reshape((28, 28)),
            cmap="gray_r",
            interpolation="nearest",
        )
        plt.xticks([])
        plt.yticks([])


def __plot_confusion_matrix(
        cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], 5)
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def __train_random_forest(x_train, y_train, x_test, y_test):
    parameters = {
        "n_estimators": [10, 20, 40, 60, 80, 100, 120, 140, 160],
        "max_features": ["auto", 15, 28, 50],
    }

    clf_rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    rf = GridSearchCV(clf_rf, parameters, n_jobs=-1)
    rf.fit(x_train, y_train)

    y_pred_rf = rf.predict(x_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("Random forest accuracy: ", acc_rf)
    return rf


def main():
    keras_backend.set_image_data_format("channels_first")

    # load the data
    cat = np.load("data/cat.npy")
    sheep = np.load("data/sheep.npy")

    cat = __add_columns_with_labels(cat, 0)
    sheep = __add_columns_with_labels(sheep, 1)

    __plot_samples(cat, title="Cat")
    __plot_samples(sheep, title="Sheep")

    # merge the cat and sheep arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
    x = np.concatenate((cat[:5000, :-1], sheep[:5000, :-1]), axis=0).astype(
        "float32"
    )  # all columns but the last
    y = np.concatenate((cat[:5000, -1], sheep[:5000, -1]), axis=0).astype(
        "float32"
    )  # the last column

    # 50:50 train/test split (divide by 255 to obtain normalized values between 0 and 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x / 255.0, y, test_size=0.5, random_state=0
    )

    rf = __train_random_forest(x_train, y_train, x_test, y_test)

    rf.predict()


if __name__ == "__main__":
    main()
