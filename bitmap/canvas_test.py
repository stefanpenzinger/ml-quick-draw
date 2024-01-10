import json
import os
import pickle
import tkinter as tk
import tensorflow as tf
from datetime import datetime
from tkinter import messagebox

import numpy as np
import constants
from PIL import Image, ImageTk, ImageGrab

# from rdp import rdp

CANVAS_SIZE = 256
BITMAP_SIZE = 28
TITLE = "ML Quick Draw"

root = tk.Tk()
root.title(TITLE)

example_file_options = os.listdir("./data")
model_options = ["knn", "rf", "cnn"]
drawing_file_options = []

clicked_example_item = tk.StringVar()
clicked_model_item = tk.StringVar()
clicked_drawing_item = tk.StringVar()

clicked_example_item.set("Choose a drawing option")
clicked_model_item.set(model_options[0])
clicked_drawing_item.set("Choose a drawing to predict")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", highlightthickness=0)
example_picture_label = tk.Label(root, image=None)

input_coordinates = []
output_coordinates = []
stroke_number = 0


def __load_example_drawing_options__():
    options = []

    for option in example_file_options:
        options.append(option.replace(".npy", ""))

    print(type(options))
    return options


def __load_prediction_drawing_options__():
    global drawing_file_options
    drawing_file_options = os.listdir("../drawings")


def __prepare_new_stroke__(event):
    input_coordinates.append([])
    input_coordinates[stroke_number].append((event.x, event.y))

    output_coordinates.append([])
    output_coordinates[stroke_number].append([])
    output_coordinates[stroke_number].append([])


def __draw__(event):
    color = "white"
    x, y = event.x, event.y
    input_coordinates[stroke_number].append((x, y))
    x1, y1 = x - 1, y - 1
    x2, y2 = x + 1, y + 1
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color, width=6)


def update_image_on_dropdown_change(var, index, mode):
    data_set = np.load("data/" + clicked_example_item.get() + ".npy")
    random_index = np.random.randint(0, len(data_set))
    bitmap = data_set[random_index].reshape((28, 28))

    new_bitmap_image = Image.fromarray(bitmap)
    scaled_image = new_bitmap_image.resize((56, 56), resample=Image.LANCZOS)
    new_photo_image = ImageTk.PhotoImage(scaled_image)

    # Update the PhotoImage object
    global photo_image  # Use a global variable to persist the updated PhotoImage
    photo_image = new_photo_image

    # Repack the label to display the updated image
    example_picture_label.config(image=photo_image)


clicked_example_item.trace_add('write', update_image_on_dropdown_change)


def __save_drawing_simplified__():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"drawings/{timestamp}-simplified-picture.ndjson"
    output_coordinates_json = {"drawing": output_coordinates}

    with open(file_name, "w") as file_stream:
        json.dump(output_coordinates_json, file_stream)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


def __save_drawing_bitmap__():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"../drawings/{timestamp}-{clicked_example_item.get()}-bitmap-picture.npy"

    canvas.update()
    image = ImageGrab.grab(bbox=(
        canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_rootx() + canvas.winfo_width(),
        canvas.winfo_rooty() + canvas.winfo_height(),))

    image.point(lambda x: int(x * 1.5))

    resized_image = image.resize((BITMAP_SIZE, BITMAP_SIZE), Image.LANCZOS)
    gray_scale_image = np.mean(np.array(resized_image), axis=2)
    gray_scale_image_flattened = gray_scale_image.flatten()

    np.save(file_name, gray_scale_image_flattened)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


def main():
    __load_prediction_drawing_options__()
    example_dropdown_items = __load_example_drawing_options__()

    # __add_dropdown_to_canvas__(example_dropdown_items, clicked_example_item)
    # __add_dropdown_to_canvas__(model_options, clicked_model_item)
    __add_buttons_to_canvas__()
    __init_canvas__()
    __add_dropdown_to_canvas__(drawing_file_options, clicked_drawing_item)
    __bind_user_actions__()

    root.mainloop()


def __init_canvas__():
    canvas.grid(row=1, column=1, sticky="w")


def __add_dropdown_to_canvas__(dropdown_items: list, clicked_view_model):
    if len(dropdown_items) > 0:
        dropdown = tk.OptionMenu(root, clicked_view_model, *dropdown_items)
        dropdown.grid(row=0, column=0, sticky="w")
        # example_picture_label.pack(side=tk.LEFT)

        reload_from_disk_button = tk.Button(root, text="Reload from Disk", command=__reload_from_disk__)
        reload_from_disk_button.grid(row=0, column=1, sticky="w")

        prediction_button = tk.Button(root, text="Predict", command=__predict__)
        prediction_button.grid(row=0, column=2, sticky="w")


def __add_buttons_to_canvas__():
    clear_button = tk.Button(root, text="Clear", command=__clear_canvas__)
    clear_button.grid(row=2, column=0, sticky="w")

    save_button = tk.Button(root, text="Save Simplified", command=__save_drawing_simplified__)
    save_button.config(state=tk.DISABLED)
    save_button.grid(row=2, column=1, sticky="w")

    save_button = tk.Button(root, text="Save Bitmap", command=__save_drawing_bitmap__)
    save_button.grid(row=2, column=2, sticky="w")


def __predict__():
    data_to_predict = np.load("../drawings/" + clicked_drawing_item.get())
    data_to_predict = data_to_predict.reshape(1, -1)
    resized_data = data_to_predict / 255

    # model = clicked_model_item.get()

    for model in model_options:
        # predict with cnn
        if model == constants.CNN_KEY:
            # Load the model from the file
            cnn_model = tf.saved_model.save("models/cnn", tags="serve")
            reshaped_data = resized_data.reshape(resized_data.shape[0], 1, 28, 28)
            pred = cnn_model.predict(reshaped_data)
            print("CNN: " + str(pred))
        # predict with other models
        else:
            # Load the model from the file
            with open("models/" + clicked_model_item.get() + ".pkl", "rb") as f:
                other_model = pickle.load(f)

            # Make predictions on new data
            print(model + ": " + str(other_model.predict(data_to_predict)))


def __reload_from_disk__():
    __load_prediction_drawing_options__()


def __clear_canvas__():
    canvas.delete(tk.ALL)


def __bind_user_actions__():
    canvas.bind("<Button-1>", __prepare_new_stroke__)
    canvas.bind("<B1-Motion>", __draw__)  # canvas.bind("<ButtonRelease-1>", update_stroke)


if __name__ == "__main__":
    main()

# def update_stroke(event):
#     global stroke_number
#     save_simplified_stroke()
#     stroke_number += 1


# def save_simplified_stroke():
#     simplified_arr = rdp(input_coordinates[stroke_number], epsilon=2.0, algo="iter", return_mask=False)
#     print(simplified_arr)
#     for i in range(len(simplified_arr)):
#         x, y = simplified_arr[i]
#         output_coordinates[stroke_number][0].append(x)
#         output_coordinates[stroke_number][1].append(y)
#
#     print(output_coordinates)
