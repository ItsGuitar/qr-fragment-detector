import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {index}.")
        return False

    time.sleep(1)  # Add a delay to give the camera time to initialize
    ret, frame = cap.read()
    if ret:
        root = tk.Tk()
        root.title(f'Camera aTest {index}')
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        label = tk.Label(root, image=img_tk)
        label.pack()
        
        root.mainloop()
        
        cap.release()
        return True
    else:
        print(f"Error: Could not read frame from camera with index {index}.")
        cap.release()
        return False

# Test different camera indices
for i in range(10):  # Extend the range to test more indices
    if test_camera(i):
        print(f"Camera with index {i} is working.")
        break