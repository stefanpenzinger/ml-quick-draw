import json
import pandas as pd


# https://github.com/keisukeirie/quickdraw_prediction_model

def parse_simplified_drawings(file_name):
    drawings = []
    with open(file_name, 'r') as file_stream:
        for line in file_stream:
            obj = json.loads(line)
            drawings.append(obj)
    return drawings


def main():
    file_name = "data/full_simplified_piano.ndjson"
    # directory = "/data"
    # for filename in os.listdir(directory):
    #     if filename.endswith(".ndjson"):
    #         pd.read_json(filename, lines=True)
    drawings = parse_simplified_drawings(file_name)
    df = pd.DataFrame(drawings)
    print(df)
    print("# of drawings:", len(df))


if __name__ == "__main__":
    main()
