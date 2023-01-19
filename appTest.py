import cv2
from PIL import Image, ImageTk
import tkinter as tk


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(0)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(
            window,
            width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tk.Button(
            window, text="Snapshot", width=50, command=self.snapshot
        )
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            cv2.imwrite(
                "frame-" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


if __name__ == "__main__":
    CameraApp(tk.Tk(), "Tkinter and OpenCV")
