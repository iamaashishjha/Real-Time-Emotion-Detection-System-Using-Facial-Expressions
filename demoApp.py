import tkinter as tk

class CameraApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Camera App")

        self.start_button = tk.Button(self.root, text="Start Camera", command=self.start_camera)
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack()

        self.root.mainloop()
    
    def start_camera(self):
        # code to start the camera goes here
        pass
    
    def stop_camera(self):
        # code to stop the camera goes here
        pass

if __name__ == "__main__":
    app = CameraApp()
