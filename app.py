import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog, ttk
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class EmotionDetectionApp:
    def __init__(self):
        # Creating object of tk class
        self.root = tk.Tk()

        # Creating object of class VideoCapture with webcam index
        self.root.cap = cv2.VideoCapture(0)

        # Setting width and height
        width, height = 640, 480
        width_1, height_1 = 640, 480
        self.root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Setting the title, window size, background color and disabling the resizing property
        self.root.title("Real Time Emotion Detection System")
        self.root.geometry("640x480")
        self.root.resizable(False, False)

        self.root.configure(background="#F0F8FF")

        self.create_widgets()
        self.configure_grid()

    def create_widgets(self):
        # Creating tkinter variables
        self.destPath = StringVar()
        self.imagePath = StringVar()

        # command line argument
        ap = argparse.ArgumentParser()
        ap.add_argument("--mode", help="train/display")
        self.mode = ap.parse_args().mode

        self.start_emotion_button = ttk.Button(self.root, text="Start Emotion Detection", command=self.start_emotion_detection, style="my.TButton")
        self.stop_emotion_button = ttk.Button(self.root, text="Stop Emotion Detection", command=self.stop_emotion_detection, style="my.TButton")
        self.emotion_label = ttk.Label(self.root, text="Emotion: ")
        self.confidence_label = ttk.Label(self.root, text="Confidence: ")

    def configure_grid(self):
        self.start_emotion_button.grid(row=0, column=0, padx=10, pady=10)
        self.stop_emotion_button.grid(row=0, column=
