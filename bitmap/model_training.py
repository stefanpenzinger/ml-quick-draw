import glob
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as keras_backend
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SAMPLES_PER_CLASS = 5000


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
    plt.show()


def __plot_confusion_matrix(y_test, y_pred, labels, model_name):
    cm = confusion_matrix(
        y_test, y_pred
    )  # create confusion matrix over all involved classes
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels.values(),
        yticklabels=labels.values(),
        title=f"{model_name} confusion matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()


def __evaluate_model(y_test, y_pred, labels, model_name) -> float:
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} test score: ", acc)

    __plot_confusion_matrix(y_test, y_pred, labels, model_name)
    return acc


def __train_model(x_train, y_train, x_test, y_test, clf, labels, model_name):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = __evaluate_model(y_test, y_pred, labels, model_name)

    return clf, acc


def __train_random_forest(x_train, y_train, x_test, y_test, labels):
    clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=12345)
    clf_rf, acc = __train_model(x_train, y_train, x_test, y_test, clf_rf, labels, "RF")

    importance = clf_rf.feature_importances_
    importance = importance.reshape((28, 28))
    plt.matshow(importance)
    plt.title("RF pixel importance\n")

    return clf_rf, acc


def __train_knn(x_train, y_train, x_test, y_test, labels):
    clf_knn = KNeighborsClassifier(n_jobs=-1)
    return __train_model(x_train, y_train, x_test, y_test, clf_knn, labels, "KNN")


def __train_svm(x_train, y_train, x_test, y_test, labels):
    clf_svm = SVC(
        kernel="rbf", random_state=0
    )  # using the Gaussian radial basis function
    return __train_model(x_train, y_train, x_test, y_test, clf_svm, labels, "SVM")


def __train_mlp(x_train, y_train, x_test, y_test, labels):
    clf_mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100, 100), random_state=0)
    return __train_model(x_train, y_train, x_test, y_test, clf_mlp, labels, "MLP")


def __get_cnn_model(num_classes):
    model_cnn = Sequential()
    model_cnn.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation="relu"))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Conv2D(15, (3, 3), activation="relu"))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation="relu"))
    model_cnn.add(Dense(50, activation="relu"))
    model_cnn.add(Dense(num_classes, activation="softmax"))
    # Compile model
    model_cnn.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model_cnn


def __train_cnn(x_train, y_train, x_test, y_test, labels):
    # one hot encode outputs
    y_train_cnn = np_utils.to_categorical(y_train)
    y_test_cnn = np_utils.to_categorical(y_test)

    # reshape to be [samples][pixels][width][height]
    x_train_cnn = x_train.reshape(x_train.shape[0], 1, 28, 28).astype("float32")
    x_test_cnn = x_test.reshape(x_train.shape[0], 1, 28, 28).astype("float32")

    np.random.seed(0)

    num_classes = y_test_cnn.shape[1]
    clf_cnn = __get_cnn_model(num_classes)

    # Fit the model
    clf_cnn.fit(
        x_train_cnn,
        y_train_cnn,
        validation_data=(x_test_cnn, y_test_cnn),
        epochs=10,
        batch_size=200,
    )

    y_pred = clf_cnn.predict(x_test_cnn)
    y_pred = np.argmax(y_pred, axis=1)

    acc = __evaluate_model(y_test, y_pred, labels, "CNN")

    return clf_cnn, acc


def __add_columns_with_labels(df, label):
    return np.c_[df, label * np.ones(len(df))]


def __load_data(
        path: str, nr_samples_per_class: int = SAMPLES_PER_CLASS
) -> (dict[int, str], pd.DataFrame, pd.DataFrame):
    labels = {}
    bitmaps_x = []
    bitmaps_y = []

    for index, file_name in enumerate(glob.glob(f"{path}/*.npy")):
        drawing_name = file_name.split("\\")[-1].split(".")[0]
        labels[index] = drawing_name

        bitmap = np.load(file_name)
        bitmap_labeled = __add_columns_with_labels(bitmap, index)

        bitmaps_x.append(bitmap_labeled[:nr_samples_per_class, :-1])
        bitmaps_y.append(bitmap_labeled[:nr_samples_per_class, -1])

        # __plot_samples(bitmap_labeled, 5, 5, title=drawing_name)

    # merge the arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
    x = np.concatenate(bitmaps_x, axis=0).astype("float32")  # all columns but the last
    y = np.concatenate(bitmaps_y, axis=0).astype("float32")  # the last column

    return labels, x, y


def main():
    keras_backend.set_image_data_format("channels_first")

    labels, x, y = __load_data("data")

    # 50:50 train/test split (divide by 255 to obtain normalized values between 0 and 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x / 255.0, y, test_size=0.5, random_state=0
    )

    rf, acc_rf = __train_random_forest(x_train, y_train, x_test, y_test, labels)
    knn, acc_knn = __train_knn(x_train, y_train, x_test, y_test, labels)
    mlp, acc_mlp = __train_mlp(x_train, y_train, x_test, y_test, labels)
    cnn, acc_cnn = __train_cnn(x_train, y_train, x_test, y_test, labels)
    svm, acc_svm = __train_svm(x_train, y_train, x_test, y_test, labels)

    drawing = np.load("drawings/20231229-145630-bitmap-picture.npy").reshape(1, -1)
    __plot_samples(drawing, 1, 1, title="Drawing")


if __name__ == "__main__":
    main()

"""
TODO:
- Grid Search, Cross Validation, Compare Models, Different Amount of samples
"""
