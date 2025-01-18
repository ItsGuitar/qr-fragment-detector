import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import datetime
import os
import numpy as np

def find_working_camera():
    for i in range(5):  # Test the first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                return i
        cap.release()
    return None

def video_stream():
    ret, frame = cap.read()
    if ret:
        # Calculate the aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        
        # Calculate the new dimensions to fit the screen while maintaining aspect ratio
        if screen_width / screen_height > aspect_ratio:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        
        # Resize the frame to the new dimensions
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a black background image
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Calculate the position to center the frame on the background
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2
        
        # Place the resized frame on the background
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
        
        # Convert the background to ImageTk format
        img = Image.fromarray(background)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new image
        label.img_tk = img_tk  
        label.config(image=img_tk)
    root.after(50, video_stream)

def on_close():
    cap.release()
    root.quit()

def take_photo():
    save_directory = os.path.join(os.getcwd(), "camera_image")
    os.makedirs(save_directory, exist_ok=True)
    ret, frame = cap.read()
    if ret:
        # Calculate the aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        
        # Calculate the new dimensions to fit the screen while maintaining aspect ratio
        if screen_width / screen_height > aspect_ratio:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        
        # Resize the frame to the new dimensions
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a black background image
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Calculate the position to center the frame on the background
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2
        
        # Place the resized frame on the background
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
        
        # Convert the background to ImageTk format
        img = Image.fromarray(background)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Save the image
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"photo_{timestamp}.png"
        img.save(os.path.join(save_directory, filename))
        print(f"Photo saved as {filename}")

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

label = Label(root)
label.place(x=0, y=0, width=screen_width, height=screen_height)

# Create a larger button with bigger text
photo_button = tk.Button(root, text="ถ่าย", command=take_photo, width=20, height=3, font=("Helvetica", 16))
photo_button.place(x=(screen_width - 200) // 2, y=screen_height - 200)  # Adjusted y coordinate to move the button up

camera_index = find_working_camera()
if camera_index is None:
    print("Error: Could not find a working camera.")
else:
    cap = cv2.VideoCapture(camera_index)
    root.protocol("WM_DELETE_WINDOW", on_close)
    video_stream()
    root.mainloop()