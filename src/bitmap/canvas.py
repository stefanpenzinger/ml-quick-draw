import os
import pickle
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import numpy as np
import tensorflow as tf
from PIL import Image, ImageGrab

import constants

CANVAS_SIZE = 256
BITMAP_SIZE = 28
TITLE = "ML Quick Draw"

root = tk.Tk()
root.title(TITLE)

canvas = tk.Canvas(
    root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", highlightthickness=0
)
example_picture_label = tk.Label(root, image=None)

knn_value = tk.StringVar()
knn_label = tk.Label(root, textvariable=knn_value)
rf_value = tk.StringVar()
rf_label = tk.Label(root, textvariable=rf_value)
mlp_value = tk.StringVar()
mlp_label = tk.Label(root, textvariable=mlp_value)
cnn_value = tk.StringVar()
cnn_label = tk.Label(root, textvariable=cnn_value)

knn_label.grid(row=1, column=0)
rf_label.grid(row=2, column=0)
mlp_label.grid(row=3, column=0)
cnn_label.grid(row=4, column=0)

clicked_example_item = tk.StringVar()
clicked_model_item = tk.StringVar()
clicked_drawing_item = tk.StringVar()


def __load_example_drawing_options():
    loaded_options = os.listdir("data")
    adapted_options = []

    for option in loaded_options:
        adapted_options.append(option.replace(".npy", ""))

    return adapted_options


def __draw(event):
    color = "white"
    x, y = event.x, event.y
    x1, y1 = x - 1, y - 1
    x2, y2 = x + 1, y + 1
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color, width=6)


def __save_drawing_bitmap():
    drawing_name = clicked_example_item.get()

    if drawing_name == "What will you draw?":
        msg = messagebox.showerror(TITLE, "You need to specify a drawing!")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"drawings/{timestamp}-{clicked_example_item.get()}-bitmap-picture.npy"

    gray_scale_image_flattened = __create_bitmap()

    np.save(file_name, gray_scale_image_flattened)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


def __create_bitmap():
    canvas.update()
    image = ImageGrab.grab(
        bbox=(
            canvas.winfo_rootx(),
            canvas.winfo_rooty(),
            canvas.winfo_rootx() + canvas.winfo_width(),
            canvas.winfo_rooty() + canvas.winfo_height(),
        )
    )

    image.point(lambda x: int(x * 10))

    resized_image = image.resize((BITMAP_SIZE, BITMAP_SIZE), Image.LANCZOS)
    gray_scale_image = np.mean(np.array(resized_image), axis=2)
    return gray_scale_image.flatten()


def main():
    drawing_file_options = ["Predict canvas"]
    for further_option in os.listdir("drawings"):
        drawing_file_options.append(further_option)
    clicked_drawing_item.set(drawing_file_options[0])
    example_file_options = __load_example_drawing_options()
    clicked_example_item.set("What will you draw?")

    __add_buttons_to_canvas()
    __init_canvas()

    dropdown_example_drawings = tk.OptionMenu(
        root, clicked_example_item, *example_file_options
    )
    dropdown_example_drawings.grid(row=0, column=1, sticky="nsew")

    dropdown_available_drawings = tk.OptionMenu(
        root, clicked_drawing_item, *drawing_file_options
    )
    dropdown_available_drawings.grid(row=5, column=0, sticky="nsew")

    prediction_button = tk.Button(root, text="Predict", command=__predict)
    prediction_button.grid(row=5, column=2, sticky="nsew")

    __bind_user_actions()

    root.mainloop()


def __init_canvas():
    canvas.grid(row=1, column=1, rowspan=4, sticky="nsew")


def __add_buttons_to_canvas():
    clear_button = tk.Button(root, text="Clear", command=__clear_canvas)
    clear_button.grid(row=0, column=0, sticky="nsew")

    save_button = tk.Button(root, text="Save Bitmap", command=__save_drawing_bitmap)
    save_button.grid(row=0, column=2, sticky="nsew")


def __predict():
    if clicked_drawing_item.get() == "Predict canvas":
        data_to_predict = __create_bitmap()
    else:
        data_to_predict = np.load("drawings/" + clicked_drawing_item.get())
    data_to_predict = data_to_predict.reshape(1, -1)
    resized_data = data_to_predict / 255
    model_options = [
        constants.KNN_KEY,
        constants.RF_KEY,
        constants.MLP_KEY,
        constants.CNN_KEY,
    ]

    for model in model_options:
        # predict with cnn
        if model == constants.CNN_KEY:
            # Load the model from the file
            cnn_model = tf.keras.models.load_model("models/cnn")
            reshaped_data = resized_data.reshape(resized_data.shape[0], 1, 28, 28)
            pred = cnn_model.predict(reshaped_data)
            print(model + ": " + str(pred))
            __update_label(model, int(np.argmax(pred, axis=1)[0]))

        # predict with other models
        else:
            # Load the model from the file
            with open("models/" + model + ".pkl", "rb") as f:
                other_model = pickle.load(f)

            predicted_index = int(other_model.predict(data_to_predict)[0])
            # Make predictions on new data
            print(model + ": " + str(predicted_index))
            __update_label(model, predicted_index)


def __update_label(model_name, index):
    print(type(index))
    animal = __load_example_drawing_options()[index]

    if model_name == constants.KNN_KEY:
        knn_value.set(str.upper(constants.KNN_KEY) + ": " + animal)
    elif model_name == constants.RF_KEY:
        rf_value.set(str.upper(constants.RF_KEY) + ": " + animal)
    elif model_name == constants.MLP_KEY:
        mlp_value.set(str.upper(constants.MLP_KEY) + ": " + animal)
    elif model_name == constants.CNN_KEY:
        cnn_value.set(str.upper(constants.CNN_KEY) + ": " + animal)


def __clear_canvas():
    canvas.delete(tk.ALL)


def __bind_user_actions():
    canvas.bind("<B1-Motion>", __draw)


if __name__ == "__main__":
    main()
