
############################################################################
###Importing libraries  
############################################################################  
import cv2, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from tkinter import *
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from datetime import datetime


############################################################################
##Variables
############################################################################
# Define directory variables
DATA_PATH = 'data/labels.txt'
DATA_PATH_TRAIN = 'data/train'
DATA_PATH_TEST = 'data/train'
DATA_MODEL_PATH = 'model/model_new.h5'
FIG_PLOT_PATH = 'imgs/plot.png'
DATE_NOW = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
LOG_DIR = "checkpoint/logs/" + DATE_NOW
TRAINING_LOG_DIR = "training.log"
# Define variables
row, col = 48, 48
# Setting width and height
width, height = 640, 480

classes = 7
num_train = 28709
num_val = 7178
image_size = 48
batch_size = 64
num_epoch = 60

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #LOG_LEVEL == 2 FOR WARNING ONLY
# Open the file
with open(DATA_PATH, 'r') as file:
    # Read the contents of the file and split it into lines
    LINES = file.read().splitlines()

# Create an empty dictionary
LABELS = {}
# # Iterate over the lines
for LINE in LINES:
    LABELS[LINE] = LINE

INDEX = [i for i in range(len(LABELS))]
LABELS_DICTIONARY = dict(zip(INDEX, LABELS))

# print(INDEX)
# print(LABELS)
# print(LABELS_DICTIONARY)

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

################################################################


################################################################
##Tkinter Variables and Properties
################################################################
# Creating object of tk class
root = tk.Tk()
# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# Setting the title, window size, background color and disabling the resizing property
root.title("Real Time Emotion Detection System")
root.wm_iconbitmap("imgs/icon.ico")
root.geometry(str(width) + "x" + str(height))
root.resizable(True, True)
root.configure(background="#F0F8FF")
root.columnconfigure(4, weight=1)
root.rowconfigure(4, weight=1)
style = ttk.Style()
style.configure(
    "my.TButton",
    background="lightblue",
    font=("Roboto Black", 15),
    width=13, relief="solid",
    highlightthickness=2,
    highlightbackground="lightgray",
    shadow="in",
    padding=10
)
# Creating tkinter variables
destPath = StringVar()
imagePath = StringVar()
########################################################################


################################################################
def get_model(input_size, classes=7):
    # Initialising the CNN
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
            activation='relu', input_shape=(row, col, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy', metrics=['accuracy'])

    return model
################################################################


# Create the model
model = get_model((row, col, 1), classes)


################################################################
def trainModel():
    # Define data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        DATA_PATH_TRAIN,
        target_size=(row, col),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = val_datagen.flow_from_directory(
        DATA_PATH_TEST,
        target_size=(row, col),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )
    
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
    )
    plot_model_history(model_info)
    model.save_weights(DATA_MODEL_PATH)
    chekpointsCheck()
    train_loss, train_acc = model.evaluate(train_generator)
    test_loss, test_acc   = model.evaluate(validation_generator)
    print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))
################################################################


################################################################    
# plots accuracy and loss curves
def plot_model_history(model_history):
    # Variable Declarations 
    train_acc = model_history.history['accuracy']
    train_loss = model_history.history['loss']
    train_val_accuracy = model_history.history["val_accuracy"]
    train_val_loss = model_history.history["val_loss"]
    x_label = "Epoch"
    y_label = "Accuracy"
    graph_labels = ["Train", "Validation"]
    label_location = "upper left"
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    fig.set_size_inches(12, 4)
    
    # summarize history for accuracy
    axs[0].plot(
        range(1, len(train_acc) + 1),
        train_acc,
    )
    axs[0].plot(
        range(1, len(train_val_accuracy) + 1),
        train_val_accuracy,
    )
    axs[0].set_title("Model Accuracy (Train Accuracy VS Validation Accuracy)")
    axs[0].set_ylabel(y_label)
    axs[0].set_xlabel(x_label)
    
    steps = len(train_acc) / 10
    xticks = np.arange(1, len(train_acc) + 1, steps)
    xticklabels = np.arange(0, len(train_acc), step=steps)
    axs[0].set_xticks(xticks, xticklabels)
    
    axs[0].legend(graph_labels, loc=label_location)
    # summarize history for loss
    axs[1].plot(
        range(1, len(train_loss) + 1), train_loss
    )
    axs[1].plot(
        range(1, len(train_val_loss) + 1),
        train_val_loss,
    )
    axs[1].set_title("Model Loss (Training Loss VS Validation Loss)")
    axs[1].set_ylabel(y_label)
    axs[1].set_xlabel(x_label)
    steps = len(train_loss) / 10
    xticks = np.arange(1, len(train_loss) + 1, steps)
    xticklabels = np.arange(0, len(train_loss), step=steps)
    axs[1].set_xticks(xticks, xticklabels)

    axs[1].legend(graph_labels, loc=label_location)
    fig.savefig(FIG_PLOT_PATH)
    plt.show()
################################################################    


################################################################    
def chekpointsCheck():
    checkpoint = ModelCheckpoint(
        filepath=DATA_MODEL_PATH, save_best_only=True, verbose=1, mode="min", moniter="val_loss"
    )

    earlystop = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=3, verbose=1, restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=6, verbose=1, min_delta=0.0001
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    csv_logger = CSVLogger(TRAINING_LOG_DIR)
    callbacks = [checkpoint, reduce_lr, csv_logger]    
################################################################    


########################################################################
# Defining DisplayFeed() function to create necessary tkinter widgets
def DisplayFeed():
    root.cameraLabel = ttk.Label(
        root, text="Camera Feed", background="#F0F8FF", wraplength = int(width)
    )
    root.cameraLabel.grid(row=4, column=4, sticky="nsew")
    
    root.cameraLabel.place(relx=0.5, rely=0.5, anchor="center")
    
    root.CAMBTN = ttk.Button(
        root, text="Stop Camera", command=StopCam, style="my.TButton"
    )
    root.CAMBTN.grid(row=5, column=4, pady=10, padx=10, sticky="nsew")
    
    # Calling ShowFeed() function
    ShowEmotionFeed()
################################################################
    
    
########################################################################
# Defining ShowEmotionFeed() function to display webcam feed in the cameraLabel;
def ShowEmotionFeed():
    # Capturing frame by frame
    model.load_weights(DATA_MODEL_PATH)
    # model.load_model(DATA_MODEL_PATH)
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Load the Haar cascade classifier
    facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    ret, frame = root.cap.read()
    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)
        # Displaying date and time on the feed
        cv2.putText(
            frame,
            DATE_NOW,
            (20, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
        )
        # Changing the frame color from BGR to GRAY
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
            text_str = LABELS_DICTIONARY[maxindex] + " --- " + str(np.round(np.max(prediction[0])*100,1)) + "%"
            cv2.putText(
                frame,
                text_str,
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
################################################################


################################################################
# Defining StartEmotionCam() to start WEBCAM Preview
def StartEmotionCam():
    # Stop Previous Camera
    StopCam()

    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)
    # Setting width and height
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN = ttk.Button(
        root, text="Stop Camera", command=StopCam, style="my.TButton"
    )
    root.CAMBTN.grid(row=5, column=4, pady=10, padx=10, sticky="nsew")

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowEmotionFeed()    
################################################################


################################################################
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
    
    root.cameraLabel.place(relx=0.5, rely=0.5, anchor="center")    
################################################################


if mode == "train":
    trainModel()


################################################################    
if __name__ == "__main__":
    # If you want to train the same model or try other models, go for this
        DisplayFeed()
        root.mainloop()
################################################################    
