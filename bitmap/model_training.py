from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CV_COUNT = 10


@dataclass
class Model:
    clf: any
    y_test_pred: np.ndarray
    num_samples: int


def __train_scikit_model(
        x_train, y_train, x_test, clf: GridSearchCV, num_samples
) -> Model:
    clf.fit(x_train, y_train)

    print("Best training parameters: " + str(clf.best_params_))
    print("Best training score: " + str(clf.best_score_))

    y_pred = clf.predict(x_test)

    return Model(clf, y_pred, num_samples)


def train_random_forest(x_train, y_train, x_test, num_samples) -> Model:
    print("### Train Random Forest")

    parameters = {"n_estimators": [10, 20, 40, 60, 80, 100, 120, 140, 160]}

    clf_rf = RandomForestClassifier(n_jobs=-1, random_state=12345)
    rf = GridSearchCV(clf_rf, parameters, n_jobs=-1, cv=CV_COUNT)

    model = __train_scikit_model(x_train, y_train, x_test, rf, num_samples)

    importance = model.clf.best_estimator_.feature_importances_
    importance = importance.reshape((28, 28))
    plt.matshow(importance)
    plt.title("RF pixel importance\n")
    plt.show()

    return model


def train_knn(x_train, y_train, x_test, num_samples) -> Model:
    print("### Train KNN")

    parameters = {"n_neighbors": [1, 3, 5, 7, 9, 11]}

    clf_knn = KNeighborsClassifier(n_jobs=-1)
    knn = GridSearchCV(clf_knn, parameters, n_jobs=-1, cv=CV_COUNT)

    return __train_scikit_model(x_train, y_train, x_test, knn, num_samples)


def train_mlp(x_train, y_train, x_test, num_samples) -> Model:
    print("### Train MLP")

    parameters = {
        "hidden_layer_sizes": [(100,), (100, 100)],
        "alpha": list(10.0 ** -np.arange(1, 4)),
    }

    clf_mlp = MLPClassifier(random_state=0)
    mlp = GridSearchCV(clf_mlp, parameters, n_jobs=-1, cv=CV_COUNT)

    return __train_scikit_model(x_train, y_train, x_test, mlp, num_samples)


def get_cnn_model(num_classes) -> Sequential:
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


def train_cnn(x_train, y_train, x_test, y_test, num_samples) -> Model:
    print("### Train CNN")

    # one hot encode outputs
    y_train_cnn = np_utils.to_categorical(y_train)
    y_test_cnn = np_utils.to_categorical(y_test)

    # reshape to be [samples][pixels][width][height]
    x_train_cnn = x_train.reshape(x_train.shape[0], 1, 28, 28).astype("float32")
    x_test_cnn = x_test.reshape(x_train.shape[0], 1, 28, 28).astype("float32")

    np.random.seed(0)

    num_classes = y_test_cnn.shape[1]
    clf_cnn = get_cnn_model(num_classes)

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

    return Model(clf_cnn, y_pred, num_samples)
