import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk, ImageGrab

# from rdp import rdp

CANVAS_SIZE = 256
BITMAP_SIZE = 28
TITLE = "ML Quick Draw"

root = tk.Tk()
root.title(TITLE)

clicked_dropdown_item = tk.StringVar()

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", highlightthickness=0)
example_picture_label = tk.Label(root, image=None)

drawing_file_names = os.listdir("./data")
input_coordinates = []
output_coordinates = []
stroke_number = 0


def __load_drawing_options__():
    options = []

    for option in drawing_file_names:
        options.append(option.replace(".npy", ""))

    print(type(options))
    return options


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
    data_set = np.load("data/" + clicked_dropdown_item.get() + ".npy")
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


def __save_drawing_simplified__():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"drawings/{timestamp}-simplified-picture.ndjson"
    output_coordinates_json = {"drawing": output_coordinates}

    with open(file_name, "w") as file_stream:
        json.dump(output_coordinates_json, file_stream)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


def __save_drawing_bitmap__():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"../drawings/{timestamp}-{clicked_dropdown_item.get()}-bitmap-picture.npy"

    canvas.update()
    image = ImageGrab.grab(bbox=(
        canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_rootx() + canvas.winfo_width(),
        canvas.winfo_rooty() + canvas.winfo_height(),))

    image.point(lambda x: int(x*1.5))

    resized_image = image.resize((BITMAP_SIZE, BITMAP_SIZE), Image.LANCZOS)
    gray_scale_image = np.mean(np.array(resized_image), axis=2)
    gray_scale_image_flattened = gray_scale_image.flatten()

    np.save(file_name, gray_scale_image_flattened)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


def main():
    dropdown_items = __load_drawing_options__()

    __add_dropdown_to_canvas__(dropdown_items)
    __init_canvas__()
    __add_buttons_to_canvas__()
    __bind_user_actions__()

    root.mainloop()


def __init_canvas__():
    canvas.pack(expand=tk.YES, fill=tk.BOTH)


def __add_dropdown_to_canvas__(dropdown_items: list):
    if len(dropdown_items) > 0:
        clicked_dropdown_item.set("Choose a drawing option")
        dropdown = tk.OptionMenu(root, clicked_dropdown_item, *dropdown_items)
        dropdown.pack(side=tk.LEFT)
        example_picture_label.pack(side=tk.LEFT)
        clicked_dropdown_item.trace_add('write', update_image_on_dropdown_change)


def __add_buttons_to_canvas__():
    clear_button = tk.Button(root, text="Clear", command=__clear_canvas__)
    clear_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=20, pady=10)

    save_button = tk.Button(root, text="Save Simplified", command=__save_drawing_simplified__)
    save_button.config(state=tk.DISABLED)
    save_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=20, pady=10)

    save_button = tk.Button(root, text="Save Bitmap", command=__save_drawing_bitmap__)
    save_button.pack(side=tk.RIGHT, fill=tk.BOTH, padx=20, pady=10)


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
