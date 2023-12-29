import json
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from PIL import ImageGrab, ImageTk
import numpy as np

from rdp import rdp

CANVAS_SIZE = 256
TITLE = "ML Quick Draw"

root = tk.Tk()
root.title(TITLE)

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
canvas.pack()

input_coordinates = []
output_coordinates = []
stroke_number = 0


def prepare_new_stroke(event):
    input_coordinates.append([])
    input_coordinates[stroke_number].append((event.x, event.y))

    output_coordinates.append([])
    output_coordinates[stroke_number].append([])
    output_coordinates[stroke_number].append([])


def draw(event):
    color = "black"
    x, y = event.x, event.y
    input_coordinates[stroke_number].append((x, y))
    x1, y1 = x - 1, y - 1
    x2, y2 = x + 1, y + 1
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)


def update_stroke(event):
    global stroke_number
    save_simplified_stroke()
    stroke_number += 1


def save_simplified_stroke():
    simplified_arr = rdp(input_coordinates[stroke_number], epsilon=2.0, algo="iter", return_mask=False)
    print(simplified_arr)
    for i in range(len(simplified_arr)):
        x, y = simplified_arr[i]
        output_coordinates[stroke_number][0].append(x)
        output_coordinates[stroke_number][1].append(y)

    print(output_coordinates)


def save_drawing_simplified():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"drawings/{timestamp}-simplified-picture.ndjson"
    output_coordinates_json = {"drawing": output_coordinates}

    with open(file_name, "w") as file_stream:
        json.dump(output_coordinates_json, file_stream)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


def save_drawing_bitmap():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"drawings/{timestamp}-bitmap-picture.npy"

    canvas.update()
    image = ImageGrab.grab(bbox=(canvas.winfo_rootx(), canvas.winfo_rooty(),
                                 canvas.winfo_rootx() + canvas.winfo_width(),
                                 canvas.winfo_rooty() + canvas.winfo_height()))
    
    canvas_data = np.array(image)
    np.save(file_name, canvas_data)


save_button = tk.Button(root, text="Save Simplified", command=save_drawing_simplified)
save_button.pack(pady=10)

save_button = tk.Button(root, text="Save Bitmap", command=save_drawing_bitmap)
save_button.pack(pady=10)

canvas.bind('<Button-1>', prepare_new_stroke)
canvas.bind('<B1-Motion>', draw)
canvas.bind('<ButtonRelease-1>', update_stroke)

root.mainloop()
