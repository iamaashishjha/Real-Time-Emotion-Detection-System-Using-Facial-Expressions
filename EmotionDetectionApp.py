import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import  ttk, *
from PIL import Image, ImageTk
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# Creating object of tk class
root = tk.Tk()

# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)

# Setting width and height
width, height = 640, 480
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Setting the title, window size, background color and disabling the resizing property
root.title("Real Time Emotion Detection System")
root.wm_iconbitmap("imgs/icon.png")
root.geometry("640x480")
root.resizable(False, False)

root.configure(background="#F0F8FF")

root.columnconfigure(4, weight=1)
root.rowconfigure(4, weight=1)

style = ttk.Style()
style.configure(
    "my.TButton", background="lightblue", font=("Roboto Black", 15), width=13
)

# Creating tkinter variables
destPath = StringVar()
imagePath = StringVar()

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Define data generators
train_dir = "data/train"
val_dir = "data/test"

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
)

# Create the model
model = Sequential()



# Defining CreateWidgets() function to create necessary tkinter widgets
def createWidgets():
    root.cameraLabel = ttk.Label(
        root, text="Camera Feed", background="steelblue", padding=10
    )
    root.cameraLabel.grid(row=4, column=4, sticky="nsew")
    root.CAMBTN = ttk.Button(
        root, text="Stop Camera", command=StopCam, style="my.TButton"
    )
    root.CAMBTN.grid(row=5, column=4, pady=10, padx=10, sticky="nsew")
    
    # Calling ShowFeed() function
    ShowEmotionFeed()

# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()
    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)
        # Displaying date and time on the feed
        cv2.putText(
            frame,
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            (20, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
        )
        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)
        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image=videoImg)
        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)
        # Keeping a reference
        root.cameraLabel.imgtk = imgtk
        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image="")

# Defining ShowEmotionFeed() function to display webcam feed in the cameraLabel;
def ShowEmotionFeed():
    # Capturing frame by frame
    model.load_weights("model.h5")

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }

    ret, frame = root.cap.read()

    # Load the Haar cascade classifier
    facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)
        # Displaying date and time on the feed
        cv2.putText(
            frame,
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            (20, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
        )
        # Changing the frame color from BGR to RGB
        # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(cv2image, scaleFactor=1.3, minNeighbors=5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = cv2image[y : y + h, x : x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(
                frame,
                emotion_dict[maxindex],
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        # Display the resulting frame
        videoImg = Image.fromarray(frame)
        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image=videoImg)
        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)
        # Keeping a reference
        root.cameraLabel.imgtk = imgtk
        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowEmotionFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image="")

# Defining StopCam() to stop WEBCAM Preview
def StopCam():
    # Stopping the camera using release() method of cv2.VideoCapture()
    root.cap.release()
    screen_width = root.winfo_screenwidth()
    width, height = root.geometry().split("x")

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(
        root, text="Open Camera", command=StartEmotionCam, style="my.TButton"
    )
    root.CAMBTN.grid(row=5, column=4, padx=10, pady=10, sticky="nsew")
    
    # Displaying text message in the camera label
    root.cameraLabel.config(
        text="Camera Switched Off", font=("Roboto Black", 50), padding=10, background="#F0F8FF", justify="center", wraplength = int(width)
    )
    
    # root.cameraLabel.grid(row=0, column=0, sticky="nsew")
    # root.cameraLabel.grid(row=4, column=4, sticky="nsew")
    # root.cameraLabel.pack(fill="both", expand=True)
    root.cameraLabel.place(relx=0.5, rely=0.5, anchor="center")

# Defining StartCam() to start WEBCAM Preview
def StartCam():
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    # Setting width and height
    # width, height = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(
        root, text="STOP CAMERA", command=StopCam, style="my.TButton"
    )
    root.CAMBTN.grid(row=5, column=4, pady=10, padx=10, sticky="nsew")

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFeed()


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Release the webcam and close all windows when the left mouse button is clicked
        root.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    createWidgets()
    root.mainloop()
