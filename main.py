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
from tkinter import *
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# Creating object of tk class
root = tk.Tk()

# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)
# bg_color = "#F0F8FF"
# bg_color-hover = "#43C6DB"
# Setting width and height
width, height = 980, 530
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Setting the title, window size, background color and disabling the resizing property
root.title("Real Time Emotion Detection System")
root.wm_iconbitmap("imgs/icon.ico")
root.geometry(str(width) + "x" + str(height))
root.resizable(False, False)

root.configure(background="#F0F8FF")

root.columnconfigure(4, weight=1)
root.rowconfigure(4, weight=1)

# style = ttk.Style()
button1_style = ttk.Style()
button2_style = ttk.Style()
button3_style = ttk.Style()
button1_style.configure("B1.TButton", font=("Roboto Black", 15), background="#F0F8FF", width=13, padding=10)
button2_style.configure("B2.TButton", font=("Roboto Black", 25), background="#F0F8FF", foreground="Red", width=13, padding=10)
button3_style.configure("B3.TButton", font=("Roboto Black", 50), padding=10, justify="center", wraplength = int(width))

# Creating a new frame to hold the buttons
button_frame = ttk.Frame(root)
button_frame.grid(row=0, column=1, rowspan=6, sticky="nsew")
button_frame.grid_rowconfigure(0, weight=1)

# Creating a new frame to hold the camera feed
camera_frame = ttk.Frame(root)
camera_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")

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


# Defining CreateWidgets() function to create necessary tkinter widgets
def createWidgets():
    # Calling ShowFeed() function
    ShowEmotionFeed()

# Defining StartCam() to start WEBCAM Preview
def StartCam():
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    # Setting width and height
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFeed()
    
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
    
# Defining StartEmotionCam() to start WEBCAM Preview
def StartFaceCam():
    # Stop Previous Camera
    StopCam()
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    # Setting width and height
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFaceFeed()
    
# Defining ShowFaceFeed() function to display webcam feed in the cameraLabel;
def ShowFaceFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()
    # Load the Haar cascade classifier
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
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
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = classifier.detectMultiScale(cv2image)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        videoImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image=videoImg)
        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)
        # Keeping a reference
        root.cameraLabel.imgtk = imgtk
        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFaceFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image="")

# Defining StartEmotionCam() to start WEBCAM Preview
def StartEmotionCam():
    # Stop Previous Camera
    StopCam()
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    # Setting width and height
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Creating the camera label and placing it in the camera frame
    root.cameraLabel = ttk.Label(
        camera_frame, text="Camera Feed", style="B3.TButton"
    )
    root.cameraLabel.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the Stop Camera button and placing it in the button frame
    root.STPCAMBTN = ttk.Button(
        button_frame, text="Stop Camera", command=StopCam, style="B2.TButton", state=ACTIVE
    )
    root.STPCAMBTN.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the Emotion Cam button and placing it in the button frame
    root.EMTNCAMBTN = ttk.Button(
        button_frame, text="Emotion Cam", command=StartEmotionCam, style="B1.TButton", state=DISABLED
    )
    root.EMTNCAMBTN.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the CAMBTN to display accordingly
    root.FCCAMBTN = ttk.Button(
        button_frame, text="Face Only", command=StartEmotionCam, style="B1.TButton", state=ACTIVE
    )
    root.FCCAMBTN.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the TRNBTN to display accordingly
    root.TRNBTN = ttk.Button(
        button_frame, text="Camera Only", command=StartFaceCam, style="B1.TButton", state=ACTIVE
    )
    root.TRNBTN.grid(row=3, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(
        button_frame, text="Camera Only", command=StartFaceCam, style="B1.TButton", state=ACTIVE
    )
    root.CAMBTN.grid(row=4, column=0, pady=10, padx=10, sticky="nsew")
    
    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowEmotionFeed()
    
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
        videoImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
    # Creating a new frame to hold the camera feed
    camera_frame = ttk.Frame(root)
    camera_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")
    
    # Displaying text message in the camera label
    root.cameraLabel = ttk.Label(
        camera_frame, text="Camera Switched Off", style="B3.TButton"
    )
    root.cameraLabel.grid(row=0, column=0, rowspan=6, pady=10, padx=10, sticky="nsew")
    
    root.cameraLabel.place(relx=0.5, rely=0.5, anchor="center")
    
    # Creating the camera label and placing it in the camera frame
    root.cameraLabel = ttk.Label(
        camera_frame, text="Camera Feed", style="B3.TButton"
    )
    root.cameraLabel.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the Stop Camera button and placing it in the button frame
    root.STPCAMBTN = ttk.Button(
        button_frame, text="Stop Camera", command=StopCam, style="B2.TButton", state=DISABLED
    )
    root.STPCAMBTN.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the Emotion Cam button and placing it in the button frame
    root.EMTNCAMBTN = ttk.Button(
        button_frame, text="Emotion Cam", command=StartEmotionCam, style="B1.TButton", state=ACTIVE
    )
    root.EMTNCAMBTN.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the CAMBTN to display accordingly
    root.FCCAMBTN = ttk.Button(
        button_frame, text="Face Only", command=StartEmotionCam, style="B1.TButton", state=ACTIVE
    )
    root.FCCAMBTN.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the TRNBTN to display accordingly
    root.TRNBTN = ttk.Button(
        button_frame, text="Camera Only", command=StartFaceCam, style="B1.TButton", state=ACTIVE
    )
    root.TRNBTN.grid(row=3, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(
        button_frame, text="Camera Only", command=StartFaceCam, style="B1.TButton", state=ACTIVE
    )
    root.CAMBTN.grid(row=4, column=0, pady=10, padx=10, sticky="nsew")

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Release the webcam and close all windows when the left mouse button is clicked
        root.cap.release()
        cv2.destroyAllWindows()

def DisplaySetup():
    root.columnconfigure(1, weight=1)
    
    # Creating the camera label and placing it in the camera frame
    root.cameraLabel = ttk.Label(
        camera_frame, text="Camera Feed", style="B3.TButton"
    )
    root.cameraLabel.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the Stop Camera button and placing it in the button frame
    root.STPCAMBTN = ttk.Button(
        button_frame, text="Stop Camera", command=StopCam, style="B2.TButton", state=ACTIVE
    )
    root.STPCAMBTN.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the Emotion Cam button and placing it in the button frame
    root.EMTNCAMBTN = ttk.Button(
        button_frame, text="Emotion Cam", command=StartEmotionCam, style="B1.TButton", state=DISABLED
    )
    root.EMTNCAMBTN.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the CAMBTN to display accordingly
    root.FCCAMBTN = ttk.Button(
        button_frame, text="Face Only", command=StartEmotionCam, style="B1.TButton", state=ACTIVE
    )
    root.FCCAMBTN.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the TRNBTN to display accordingly
    root.TRNBTN = ttk.Button(
        button_frame, text="Camera Only", command=StartFaceCam, style="B1.TButton", state=ACTIVE
    )
    root.TRNBTN.grid(row=3, column=0, pady=10, padx=10, sticky="nsew")
    
    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(
        button_frame, text="Camera Only", command=StartFaceCam, style="B1.TButton", state=ACTIVE
    )
    root.CAMBTN.grid(row=4, column=0, pady=10, padx=10, sticky="nsew")
    
    # Calling ShowFeed() function
    ShowEmotionFeed()


if __name__ == "__main__":
    DisplaySetup()
    # createWidgets()
    root.mainloop()
    
    
