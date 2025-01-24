import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import os
import numpy as np
import random
import string
import subprocess
import threading
import queue
import re

def find_working_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                return i
        cap.release()
    return None

def calculate_contrast_color(color):
    luminance = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

def video_stream():
    if not capturing.is_set() and show_camera_feed:
        ret, frame = cap.read()
        if ret:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            if screen_width / screen_height > aspect_ratio:
                new_height = screen_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = screen_width
                new_height = int(new_width / aspect_ratio)

            frame = cv2.resize(frame, (new_width, new_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            x_offset = (screen_width - new_width) // 2
            y_offset = (screen_height - new_height) // 2
            background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame

            avg_color = np.mean(frame, axis=(0, 1)).astype(int)
            contrast_color = calculate_contrast_color(avg_color)

            square_size = min(new_width, new_height) // 3
            top_left = (
                x_offset + (new_width - square_size) // 2,
                y_offset + (new_height - square_size) // 2 - 50
            )
            bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
            cv2.rectangle(background, top_left, bottom_right, contrast_color, 3)

            img = Image.fromarray(background)
            img_tk = ImageTk.PhotoImage(image=img)
            label.img_tk = img_tk
            label.config(image=img_tk, bg="black")
    root.after(50, video_stream)

def on_close():
    capturing.set()
    cap.release()
    root.quit()

def take_photo():
    global show_camera_feed
    capturing.set()
    show_camera_feed = False
    label.config(image='', bg="black")

    save_directory = os.path.join(os.getcwd(), "camera_image")
    os.makedirs(save_directory, exist_ok=True)
    photo_path = os.path.join(save_directory, "photo.png")

    if os.path.exists(photo_path):
        random_char = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        new_name = f"photo_{random_char}.png"
        os.rename(photo_path, os.path.join(save_directory, new_name))
        print(f"Existing photo renamed to {new_name}")

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(photo_path, frame)
        print("Photo saved as photo.png")

        aspect_ratio = frame.shape[1] / frame.shape[0]
        if screen_width / screen_height > aspect_ratio:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame

        avg_color = np.mean(frame, axis=(0, 1)).astype(int)
        contrast_color = calculate_contrast_color(avg_color)

        square_size = min(new_width, new_height) // 3
        top_left = (
            x_offset + (new_width - square_size) // 2,
            y_offset + (new_height - square_size) // 2 - 50
        )
        bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
        cv2.rectangle(background, top_left, bottom_right, contrast_color, 3)

        img = Image.fromarray(background)
        img_tk = ImageTk.PhotoImage(image=img)
        label.img_tk = img_tk
        label.config(image=img_tk, bg="black")

        photo_button.place_forget()
        loading_label.place(x=0, y=0, width=screen_width, height=screen_height)
        root.update()

        thread = threading.Thread(
            target=run_training_and_get_result,
            args=(result_queue,)
        )
        thread.start()

def run_training_and_get_result(q):
    process = subprocess.Popen(["python", "training.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        output = stdout.decode()
        match_message = ""
        for line in output.splitlines():
            if "Best overall match found" in line:
                match_message = line
                break
        if match_message:
            q.put(match_message)
        else:
            q.put("No match message found in output.")
    else:
        q.put(f"Error: {stderr.decode()}")

def check_training_result():
    try:
        message = result_queue.get_nowait()
        show_match_message(message)
    except queue.Empty:
        pass
    root.after(100, check_training_result)

def show_match_message(message):
    global show_camera_feed
    show_camera_feed = False
    label.config(image='', bg="black")

    match_label.config(text=message, font=("Helvetica", 12), bg="black", fg="white")
    match_label.place(x=10, y=screen_height - 70, anchor='sw')
    countdown_label.config(bg="black", fg="white")
    countdown_label.place(x=screen_width // 2, y=(screen_height // 2), anchor='center')

    info_label.config(text="หากรูปไม่ตรง โปรดมองข้าม", font=("Helvetica", 36), bg="black", fg="white")
    info_label.place(x=screen_width // 2, y=100, anchor='center')

    match = re.search(r'Best overall match found in test image (\d+) with match percentage (\d+\.\d+)%', message)
    if match:
        best_test_image = match.group(1)

        # Read the rotation angle from the file
        rotation_angle = 0
        try:
            with open(os.path.join("output", "best_overall_match.txt"), 'r') as f:
                for line in f:
                    if "Rotation Angle" in line:
                        rotation_angle = int(re.search(r'(\d+)', line).group(1))
                        break
        except FileNotFoundError:
            print("Rotation angle file not found.")

        best_overall_image_path = os.path.join("output", "best_overall_image.jpg")
        if os.path.exists(best_overall_image_path):
            best_overall_image = Image.open(best_overall_image_path)
            best_overall_image = best_overall_image.resize((200, 200), Image.Resampling.LANCZOS)
            best_overall_image_tk = ImageTk.PhotoImage(best_overall_image)
            enclosed_area_label.config(image=best_overall_image_tk)
            enclosed_area_label.image = best_overall_image_tk
            enclosed_area_label.place(x=(screen_width // 2) - 270, y=(screen_height // 2) - 200)

        test_image_path = os.path.join("test_image", f"test{best_test_image}.jpg")
        if os.path.exists(test_image_path):
            test_image_img = Image.open(test_image_path)
            test_image_img = test_image_img.resize((200, 200), Image.Resampling.LANCZOS)
            test_image_img_tk = ImageTk.PhotoImage(test_image_img)
            test_image_label.config(image=test_image_img_tk)
            test_image_label.image = test_image_img_tk
            test_image_label.place(x=(screen_width // 2) + 70, y=(screen_height // 2) - 200)

        rotation_label.config(text=f"หมุน\n{rotation_angle}°\nตามเข็ม", font=("Helvetica", 16), bg="black", fg="white")
        rotation_label.place(x=screen_width // 2, y=(screen_height // 2) - 150, anchor='center')

        # Display the gridx.png below the countdown timer, at the same 200×200 size
        grid_image_path = os.path.join("test_image", "location", f"grid{best_test_image}.png")
        if os.path.exists(grid_image_path):
            grid_image = Image.open(grid_image_path)
            grid_image = grid_image.resize((200, 200), Image.Resampling.LANCZOS)
            grid_image_tk = ImageTk.PhotoImage(grid_image)
            grid_image_label.config(image=grid_image_tk)
            grid_image_label.image = grid_image_tk
            grid_image_label.place(x=screen_width // 2, y=(screen_height // 2) + 150, anchor='center')

    root.update()
    countdown(10)

def countdown(seconds):
    global show_camera_feed
    loading_label.place_forget()
    root.config(bg="black")
    if seconds > 0:
        countdown_label.config(text=f"{seconds}", font=("Helvetica", 24), bg="black", fg="white")
        root.after(1000, countdown, seconds-1)
    else:
        match_label.place_forget()
        countdown_label.place_forget()
        enclosed_area_label.place_forget()
        test_image_label.place_forget()
        info_label.place_forget()
        rotation_label.place_forget()
        grid_image_label.place_forget()
        capturing.clear()
        show_camera_feed = True
        root.config(bg="white")
        label.config(bg="white", image='')
        photo_button.place(x=(screen_width - 200) // 2, y=screen_height - 200)
        root.update()

import queue
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

capturing = threading.Event()
result_queue = queue.Queue()
show_camera_feed = True

label = Label(root, bg="black")
label.place(x=0, y=0, width=screen_width, height=screen_height)

loading_label = Label(root, text="Loading...", font=("Helvetica", 32), bg="black", fg="white")
loading_label.place_forget()

match_label = Label(root, text="", font=("Helvetica", 12), bg="black", fg="white")
match_label.place_forget()

countdown_label = Label(root, text="", font=("Helvetica", 24), bg="black", fg="white")
countdown_label.place_forget()

info_label = Label(root, text="", font=("Helvetica", 16), bg="black", fg="white")
info_label.place_forget()

rotation_label = Label(root, text="", font=("Helvetica", 16), bg="black", fg="white")
rotation_label.place_forget()

enclosed_area_label = Label(root)
enclosed_area_label.place_forget()

test_image_label = Label(root)
test_image_label.place_forget()

grid_image_label = Label(root)
grid_image_label.place_forget()

photo_button = tk.Button(root, text="ถ่าย", command=take_photo, width=20, height=3, font=("Helvetica", 16))
photo_button.place(x=(screen_width - 200) // 2, y=screen_height - 200)

camera_index = find_working_camera()
if camera_index is None:
    print("Error: Could not find a working camera.")
else:
    cap = cv2.VideoCapture(camera_index)
    root.protocol("WM_DELETE_WINDOW", on_close)
    video_stream()
    check_training_result()
    root.mainloop()