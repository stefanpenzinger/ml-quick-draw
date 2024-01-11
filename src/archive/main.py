import preprocessing
import model_training

# Warning: doesn't work
if __name__ == "__main__":
    df = preprocessing.read_and_preprocess("data")
    df_drew = preprocessing.read_and_preprocess("../bitmap/drawings")
    model_training.train_gallery_dependent(df, df_drew)
