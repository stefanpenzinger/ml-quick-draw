import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# Function to plot 28x28 pixel drawings that are stored in a numpy array.
# Specify how many rows and cols of pictures to display (default 4x5).
# If the array contains fewer images than subplots selected, surplus subplots remain empty.
def plot_samples(input_array, rows=4, cols=5, title=""):
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


def plot_confusion_matrix(y_test, y_pred, labels, model_name):
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


def plot_acc_scores(acc_scores, max_samples):
    acc_scores.transpose().plot(
        xticks=list(acc_scores),
        xlim=(0, max_samples),
        marker="o",
        title="Classification accuracy of algorithms\n",
    )
    plt.xlabel("# of training examples")
    plt.ylabel("accuracy [%]")
    plt.show()
