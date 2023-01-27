import cv2, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array

from tkinter import *
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #LOG_LEVEL == 2 FOR WARNING ONLY

DATA_PATH = 'data/test'
CATEGORIES = os.listdir(DATA_PATH)
LABELS = [i for i in range(len(CATEGORIES))]
LABELS_DICTIONARY = dict(zip(CATEGORIES, LABELS))

print(CATEGORIES)
print(LABELS)
print(LABELS_DICTIONARY)


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

# Define data generators
train_dir = "data/train"
val_dir = "data/test"

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = train_datagen

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

# Defining ShowEmotionFeed() function to display webcam feed in the cameraLabel;
def ShowEmotionFeed():
    # Capturing frame by frame
    model.load_weights("model.h5")
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)

    # Setting width and height
    width, height = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Load the Haar cascade classifier
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    res, frame = root.cap.read()
    if res:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)
        # Displaying date and time on the feed
        height, width , channel = frame.shape
        sub_img = frame[0:int(height/6),0:int(width)]

        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1
        FONT_THICKNESS = 2
        LABEL_COLOR = (0, 0, 0)
        LABEL_COLOR = (0, 0, 0)
        LABEL_TITLE = "Real Time Emotion Detection"
        LABEL_DIMENSION = cv2.getTextSize(LABEL_TITLE,FONT,FONT_SCALE,FONT_THICKNESS)[0]
        textX = int((res.shape[1] - LABEL_DIMENSION[0]) / 2)
        textY = int((res.shape[0] + LABEL_DIMENSION[1]) / 2)
        # cv2.putText(frame, LABEL_TITLE, (textX,textY), FONT, FONT_SCALE, LABEL_COLOR, FONT_THICKNESS)
        text_width, _ = cv2.getTextSize(LABEL_TITLE, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        rect_x, rect_y = textX - 5, textY - 25
        rect_width, rect_height = text_width + 10, 30
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), LABEL_COLOR, -1)
        cv2.putText(frame, LABEL_TITLE, (textX,textY), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_prediction = emotion_detection[max_index]
            emotion_prediction_confidence = str(np.round(np.max(predictions[0])*100,1))+ "%"
            LABEL_EMOTION = 'Emotion: {}'.format(emotion_prediction)
            # LABEL_CONFIDENCE = 'Confidence: {}'.format(emotion_prediction_confidence)
            LABEL_CONFIDENCE = emotion_prediction_confidence
            
            violation_text_dimension = cv2.getTextSize(LABEL_CONFIDENCE,FONT,FONT_SCALE,FONT_THICKNESS)[0]
            # LABEL_X_AXIS = int(res.shape[1]- violation_text_dimension[0])
            # LABEL_Y_AXIS = textY+22+5
            LABEL_X_AXIS = x+20
            LABEL_Y_AXIS = y-60
            
            cv2.putText(
                frame,
                LABEL_EMOTION,
                (0,LABEL_Y_AXIS),
                FONT,
                FONT_SCALE,
                LABEL_COLOR,
                FONT_THICKNESS,
            )
            cv2.putText(
                frame,
                LABEL_CONFIDENCE,
                (LABEL_X_AXIS,LABEL_Y_AXIS),
                FONT,
                FONT_SCALE,
                LABEL_COLOR,
                FONT_THICKNESS,
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

# # Defining StartEmotionCam() to start WEBCAM Preview
# def StartEmotionCam():
#     # Stop Previous Camera
#     StopCam()

#     # Creating object of class VideoCapture with webcam index
#     root.cap = cv2.VideoCapture(0)
#     # Setting width and height
#     root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#     # Configuring the CAMBTN to display accordingly
#     root.CAMBTN = ttk.Button(
#         root, text="Stop Camera", command=StopCam, style="my.TButton"
#     )
#     root.CAMBTN.grid(row=5, column=4, pady=10, padx=10, sticky="nsew")

#     # Removing text message from the camera label
#     root.cameraLabel.config(text="")

#     # Calling the ShowFeed() Function
#     ShowEmotionFeed()
    
    
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
    
    
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Release the webcam and close all windows when the left mouse button is clicked
        root.cap.release()
        cv2.destroyAllWindows()


def trainModel():
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
    )
    plot_model_history(model_info)
    model.save_weights("model.h5")
    
    
if __name__ == "__main__":
    # DisplayFeed()
    root.mainloop()
    
    
