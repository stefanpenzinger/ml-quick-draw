from keras import backend as keras_backend
from sklearn.model_selection import train_test_split

from io_operations import load_data
from model_training import (
    train_random_forest,
    train_knn,
    train_mlp,
    train_cnn,
)
from model_evaluation import evaluate_best_model

RF_KEY = "rf"
KNN_KEY = "knn"
MLP_KEY = "mlp"
CNN_KEY = "cnn"
NUM_SAMPLES = [2000, 5000, 10000, 15000]

if __name__ == "__main__":
    keras_backend.set_image_data_format("channels_first")

    labels = {}
    trained_models = {RF_KEY: [], KNN_KEY: [], MLP_KEY: [], CNN_KEY: []}
    y_test_per_sample_size = {}

    for index, num_samples in enumerate(NUM_SAMPLES):
        print(f"#### Using {num_samples} samples")

        labels, x, y = load_data("data", num_samples, should_plot=index == 0)

        # 50:50 train/test split (divide by 255 to obtain normalized values between 0 and 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x / 255.0, y, test_size=0.5, random_state=0
        )
        y_test_per_sample_size[num_samples] = y_test

        trained_models[RF_KEY].append(
            train_random_forest(x_train, y_train, x_test, num_samples)
        )
        trained_models[KNN_KEY].append(train_knn(x_train, y_train, x_test, num_samples))
        trained_models[MLP_KEY].append(train_mlp(x_train, y_train, x_test, num_samples))
        trained_models[CNN_KEY].append(
            train_cnn(x_train, y_train, x_test, y_test, num_samples)
        )

    evaluate_best_model(labels, trained_models, y_test_per_sample_size)
