import pandas as pd
from sklearn.metrics import accuracy_score

from io_operations import save_model
from plot import plot_confusion_matrix, plot_acc_scores


def __get_test_score(y_test, y_pred, model_name) -> float:
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} test score: ", acc)
    return acc


def evaluate_best_model(labels, trained_models, y_test_per_sample_size):
    print("#### Evaluating best model")
    acc_scores = pd.DataFrame()
    for model_type, models in trained_models.items():
        print(f"### {model_type}")
        best_acc = 0
        best_model = None

        for model in models:
            model_name = f"{model_type} ({model.num_samples} samples)"
            acc = __get_test_score(
                y_test_per_sample_size[model.num_samples], model.y_test_pred, model_name
            )
            acc_scores.at[model_type, model.num_samples] = acc

            if acc > best_acc:
                best_acc = acc
                best_model = model

        if best_model is not None:
            plot_confusion_matrix(
                y_test_per_sample_size[best_model.num_samples],
                best_model.y_test_pred,
                labels,
                f"{model_type} ({best_model.num_samples})",
            )
            save_model(best_model.clf, model_type)

    plot_acc_scores(acc_scores, max(y_test_per_sample_size.keys()))
