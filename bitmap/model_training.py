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
    plt.show()


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
    plt.show()


def __train_random_forest(x_train, y_train, x_test, y_test):
    clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=12345)
    clf_rf.fit(x_train, y_train)

    y_pred_rf = clf_rf.predict(x_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("Random forest accuracy: ", acc_rf)
    return clf_rf


def __train_knn(x_train, y_train, x_test, y_test):
    clf_knn = KNeighborsClassifier(n_jobs=-1)
    clf_knn.fit(x_train, y_train)

    y_pred_knn = clf_knn.predict(x_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print("KNN accuracy: ", acc_knn)
    return clf_knn


def __train_svm(x_train, y_train, x_test, y_test):
    clf_svm = SVC(
        kernel="rbf", random_state=0
    )  # using the Gaussian radial basis function
    clf_svm.fit(x_train, y_train)

    y_pred_svm = clf_svm.predict(x_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print("RBF SVM accuracy: ", acc_svm)
    return clf_svm


def __train_mlp(x_train, y_train, x_test, y_test):
    clf_mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100, 100), random_state=0)
    clf_mlp.fit(x_train, y_train)

    y_pred_mlp = clf_mlp.predict(x_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    print("MLP accuracy: ", acc_mlp)
    return clf_mlp


def __train_cnn(x_train, y_train, x_test, y_test):
    # one hot encode outputs
    y_train_cnn = np_utils.to_categorical(y_train)
    y_test_cnn = np_utils.to_categorical(y_test)
    num_classes = y_test_cnn.shape[1]

    # reshape to be [samples][pixels][width][height]
    x_train_cnn = x_train.reshape(x_train.shape[0], 1, 28, 28).astype("float32")
    x_test_cnn = x_test.reshape(x_train.shape[0], 1, 28, 28).astype("float32")

    np.random.seed(0)

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

    # Fit the model
    model_cnn.fit(
        x_train_cnn,
        y_train_cnn,
        validation_data=(x_test_cnn, y_test_cnn),
        epochs=10,
        batch_size=200,
    )
    # Final evaluation of the model
    scores = model_cnn.evaluate(x_test_cnn, y_test_cnn, verbose=0)
    print("Final CNN accuracy: ", scores[1])
    return model_cnn


def main():
    keras_backend.set_image_data_format("channels_first")

    # load the data
    cat = np.load("data/cat.npy")
    sheep = np.load("data/sheep.npy")

    cat = __add_columns_with_labels(cat, 1)
    sheep = __add_columns_with_labels(sheep, 0)

    __plot_samples(cat, 1, 1, title="Cat")
    __plot_samples(sheep, 1, 1, title="Sheep")

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
    knn = __train_knn(x_train, y_train, x_test, y_test)
    svm = __train_svm(x_train, y_train, x_test, y_test)
    mlp = __train_mlp(x_train, y_train, x_test, y_test)
    cnn = __train_cnn(x_train, y_train, x_test, y_test)

    drawing = np.load("drawings/20231229-145630-bitmap-picture.npy").reshape(1, -1)
    __plot_samples(drawing, 1, 1, title="Drawing")


if __name__ == "__main__":
    main()

"""
TODO:
- Canvas - show available drawings -> Draw -> Recognize with different models (+ probability?)
- more drawings -> lower accuracy
- Grid Search, Cross Validation, ROC Curve, AUC, Confusion Matrix, Compare Models, More Animals, Describe "Features"
- Pixel Importance
- Sample number comparison (1000, 2000, 3000, 4000, 5000, etc.) measured in model performance 
"""
