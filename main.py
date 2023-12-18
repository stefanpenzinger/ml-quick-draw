import json
import math

import pandas as pd


def get_direction_of_stroke(df_row: pd.DataFrame, stroke_n: int) -> float:
    if df_row["strokeCount"] <= stroke_n:
        return 0

    x = df_row["drawing"][stroke_n][0]
    y = df_row["drawing"][stroke_n][1]

    dx = x[len(x) - 1] - x[0]
    dy = y[len(y) - 1] - y[0]

    return math.atan2(dx, dy) * 180 / math.pi


def get_datapoint_percentage_of_stroke(df_row: pd.DataFrame, stroke_n: int) -> float:
    if df_row["strokeCount"] <= stroke_n:
        return 0
    return len(df_row["drawing"][stroke_n][0]) / df_row["datapointCount"]


def feature_extraction(df_in: pd.DataFrame) -> pd.DataFrame:
    # based on https://github.com/keisukeirie/quickdraw_prediction_model
    df_in["datapointCount"] = df_in["x"].apply(lambda x: len(x))
    df_in["strokeCount"] = df_in["drawing"].apply(lambda drawing: len(drawing))

    df_in["yMax"] = df_in["y"].apply(lambda row: max(row))

    df_in["datapointPercentageStroke0"] = df_in.apply(
        lambda row: get_datapoint_percentage_of_stroke(row, 0), axis=1
    )
    df_in["datapointPercentageStroke1"] = df_in.apply(
        lambda row: get_datapoint_percentage_of_stroke(row, 1), axis=1
    )
    df_in["datapointPercentageStroke2"] = df_in.apply(
        lambda row: get_datapoint_percentage_of_stroke(row, 2), axis=1
    )

    df_in["directionStroke0"] = df_in.apply(
        lambda row: get_direction_of_stroke(row, 0), axis=1
    )
    df_in["directionStroke1"] = df_in.apply(
        lambda row: get_direction_of_stroke(row, 1), axis=1
    )
    df_in["directionStroke2"] = df_in.apply(
        lambda row: get_direction_of_stroke(row, 2), axis=1
    )

    df_in["x0"] = df_in["x"].apply(lambda x: x[0])
    df_in["y0"] = df_in["y"].apply(lambda x: x[0])
    return df_in


def extract_coordinates_from_drawing(drawing: list, coordinate_idx: int) -> list:
    values = []
    for stroke in drawing:
        values.extend(stroke[coordinate_idx])
    return values


def parse_simplified_drawings(file_name: str) -> pd.DataFrame:
    drawings = []
    with open(file_name, "r") as file_stream:
        count = 0
        for line in file_stream:
            obj = json.loads(line)
            drawings.append(obj)
            count += 1
            if count >= 1000:
                break
    df = pd.DataFrame(drawings)
    df["x"] = df["drawing"].apply(
        lambda drawing: extract_coordinates_from_drawing(drawing, 0)
    )
    df["y"] = df["drawing"].apply(
        lambda drawing: extract_coordinates_from_drawing(drawing, 1)
    )
    return df


def main():
    file_name = "data/full_simplified_piano.ndjson"
    df = parse_simplified_drawings(file_name)
    df_features = feature_extraction(df)
    print(df_features)


if __name__ == "__main__":
    main()
