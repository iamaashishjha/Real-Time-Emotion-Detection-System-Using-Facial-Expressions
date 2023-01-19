import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog, ttk
# from tkinter import ttk

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
# from tf.keras.optimizers.legacy.Optimizer import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Creating object of tk class
root = tk.Tk()

def loadDefaults():

    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)

    # Setting width and height
    width, height = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Setting the title, window size, background color and disabling the resizing property
    root.title("Real Time Emotion Detection System")
    root.geometry("663x560")
    root.resizable(False, False)

    root.configure(background = "#F0F8FF")

    style = ttk.Style()
    style.configure("my.TButton", background="lightblue", font=('Roboto Black',15), width=13)

    # Creating tkinter variables
    destPath = StringVar()
    imagePath = StringVar()

    # command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",help="train/display")
    mode = ap.parse_args().mode

    # Define data generators
    train_dir = 'data/train'
    val_dir = 'data/test'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

# Defining CreateWidgets() function to create necessary tkinter widgets
def createWidgets():
    root.columnconfigure(4, weight=1)
    root.rowconfigure(4, weight=1)

    root.cameraLabel = ttk.Label(root, text="CAMERA FEED", background="steelblue", padding=10)
    root.cameraLabel.grid(row=4, column=4, sticky="nsew")

    root.CAMBTN = ttk.Button(root, text="STOP CAMERA", command=StopCAM, style="my.TButton")
    root.CAMBTN.grid(row=5, column=4, pady=10, padx=10, sticky="nsew")

    # Calling ShowFeed() function
    ShowFeed()

# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()
    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)
        # Displaying date and time on the feed
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))
        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)
        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image = videoImg)
        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)
        # Keeping a reference
        root.cameraLabel.imgtk = imgtk
        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image='')

# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowEmotionFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()
    # if ret:
    #     # Flipping the frame vertically
    #     frame = cv2.flip(frame, 1)
    #     # Displaying date and time on the feed
    #     cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))
    #     # Changing the frame color from BGR to RGB
    #     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #     # Creating an image memory from the above frame exporting array interface
    #     videoImg = Image.fromarray(cv2image)
    #     # Creating object of PhotoImage() class to display the frame
    #     imgtk = ImageTk.PhotoImage(image = videoImg)
    #     # Configuring the label to display the frame
    #     root.cameraLabel.configure(image=imgtk)
    #     # Keeping a reference
    #     root.cameraLabel.imgtk = imgtk
    #     # Calling the function after 10 milliseconds
    #     root.cameraLabel.after(10, ShowFeed)
    # else:
    #     # Configuring the label to display the frame
    #     root.cameraLabel.configure(image='')

    model.load_weights('model.h5')
    # Capturing frame by frame
    ret, frame = root.cap.read()
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = root.cap.read()
        if not ret:
            break
            # Configuring the label to display the frame
            # root.cameraLabel.configure(image='')
        
        frame = cv2.flip(frame, 1)
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the webcam feed in the defined window
        # resizedFrame = cv2.resize(frame,(665,560),interpolation = cv2.INTER_CUBIC)
        # cv2.imshow("Webcam Feed", resizedFrame)
        videoImg = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image = videoImg)
        root.cameraLabel.configure(image=imgtk)
        root.cameraLabel.imgtk = imgtk
        root.cameraLabel.after(10, ShowFeed)

# Defining StopCAM() to stop WEBCAM Preview
def StopCAM():
    # Stopping the camera using release() method of cv2.VideoCapture()
    root.cap.release()
    root.columnconfigure(4, weight=1)
    root.rowconfigure(4, weight=1)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(root, text="START CAMERA", command=StartCAM, style="my.TButton")
    root.CAMBTN.grid(row=5, column=4, padx=10, pady=10, sticky="nsew")

    # Displaying text message in the camera label
    root.cameraLabel.config(text="OFF CAM", font=('Roboto Black',70), padding=10, justify="center")

# Defining StartCAM() to start WEBCAM Preview
def StartCAM():
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    root.columnconfigure(4, weight=1)
    root.rowconfigure(4, weight=1)
    # Setting width and height
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(root, text="STOP CAMERA", command=StopCAM, style="my.TButton")
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
    loadDefaults()
    createWidgets()
    root.mainloop()

