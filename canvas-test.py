import tkinter as tk
from tkinter import messagebox
import json
from datetime import datetime

from rdp import rdp

CANVAS_SIZE = 256
TITLE = "ML Quick Draw"

root = tk.Tk()
root.title(TITLE)

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
canvas.pack()

inputCoordinates = []
outputCoordinates = []
stroke_number = 0


def prepare_new_stroke(event):
    inputCoordinates.append([])
    inputCoordinates[stroke_number].append((event.x, event.y))

    outputCoordinates.append([])
    outputCoordinates[stroke_number].append([])
    outputCoordinates[stroke_number].append([])


def draw(event):
    color = "black"
    x, y = event.x, event.y
    inputCoordinates[stroke_number].append((x, y))
    x1, y1 = x - 1, y - 1
    x2, y2 = x + 1, y + 1
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)


def update_stroke(event):
    global stroke_number
    save_simplified_stroke()
    stroke_number += 1


def save_simplified_stroke():
    simplified_arr = rdp(inputCoordinates[stroke_number], epsilon=2.0, algo="iter", return_mask=False)
    print(simplified_arr)
    for i in range(len(simplified_arr)):
        x, y = simplified_arr[i]
        outputCoordinates[stroke_number][0].append(x)
        outputCoordinates[stroke_number][1].append(y)

    print(outputCoordinates)


def save_drawing():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"drawings/{timestamp}-picture.json"
    with open(file_name, "w") as file_stream:
        json.dump(outputCoordinates, file_stream)
    msg = messagebox.showinfo(TITLE, "Saved successfully!")


save_button = tk.Button(root, text="Save", command=save_drawing)
save_button.pack(pady=10)

canvas.bind('<Button-1>', prepare_new_stroke)
canvas.bind('<B1-Motion>', draw)
canvas.bind('<ButtonRelease-1>', update_stroke)

root.mainloop()
