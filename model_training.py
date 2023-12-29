import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def __create_confusion_matrix__(target, predicted, title):
    cm = confusion_matrix(
        target, predicted
    )  # create confusion matrix over all involved classes
    print(cm)

    sorted_target = sorted(target.unique())

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=sorted_target,
        yticklabels=sorted_target,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()


def __calculate_accuracy_score__(target, predicted, title):
    print(title + ": " + str(metrics.accuracy_score(target, predicted)))


def __train__(
        data_train, data_test, target_train, target_test, classification, df_in_drew
):
    classification.fit(data_train, target_train)

    print("Best training parameters: " + str(classification.best_params_))
    print("Best training score: " + str(classification.best_score_))

    train_predicted = classification.predict(
        data_train
    )  # let model predict output variable for all training set samples
    test_predicted = classification.predict(
        data_test
    )  # let model predict output variable for all test set samples
    drew_predicted = classification.predict(df_in_drew)
    print(drew_predicted)

    __create_confusion_matrix__(
        target_train,
        train_predicted,
        f"{classification.estimator}: Train Confusion Matrix",
    )
    __create_confusion_matrix__(
        target_test,
        test_predicted,
        f"{classification.estimator}: Test Confusion Matrix",
    )

    __calculate_accuracy_score__(target_train, train_predicted, "Train Score")
    __calculate_accuracy_score__(target_test, test_predicted, "Test Score")


def __train_knn__(data_train, data_test, target_train, target_test, df_in_drew):
    print("#### KNN")
    parameters_to_tune = {"n_neighbors": [n for n in range(1, 10)]}

    knn_classification = GridSearchCV(
        KNeighborsClassifier(), parameters_to_tune, cv=10, scoring="accuracy"
    )
    __train__(
        data_train, data_test, target_train, target_test, knn_classification, df_in_drew
    )


def __train_random_forest__(
        data_train, data_test, target_train, target_test, df_in_drew
):
    print("#### Random Forest")
    parameters_to_tune = {
        "max_depth": [5, 10, 20, 40, 80],
        "n_estimators": [1, 3, 5, 10],
    }

    forest_classification = GridSearchCV(
        RandomForestClassifier(),
        parameters_to_tune,
        cv=10,
        scoring="accuracy",
    )
    __train__(
        data_train,
        data_test,
        target_train,
        target_test,
        forest_classification,
        df_in_drew,
    )


def __train_decision_tree__(
        data_train, data_test, target_train, target_test, df_in_drew
):
    print("#### Decision Tree")
    parameters_to_tune = {
        "max_depth": [5, 10, 20, 40, 80, 120],
        "max_features": [2 ** n for n in range(0, 3)],
    }

    tree_classification = GridSearchCV(
        DecisionTreeClassifier(), parameters_to_tune, cv=10, scoring="accuracy"
    )
    __train__(
        data_train,
        data_test,
        target_train,
        target_test,
        tree_classification,
        df_in_drew,
    )


def train_gallery_dependent(df_in: pd.DataFrame, df_in_drew: pd.DataFrame):
    # train/test split
    data_train, data_test, target_train, target_test = train_test_split(
        df_in.iloc[:, 10:], df_in.iloc[:, 0], train_size=0.75, random_state=321
    )
    data_drew = df_in_drew.iloc[:, 5:]

    # __train_knn__(data_train, data_test, target_train, target_test, data_drew)
    __train_random_forest__(data_train, data_test, target_train, target_test, data_drew)
    __train_decision_tree__(data_train, data_test, target_train, target_test, data_drew)
