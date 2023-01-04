import tkinter as tk
import subprocess

def run_script():
    subprocess.run(['python', 'emotions.py', '--mode', 'display'])

app = tk.Tk()
app.title('My App')

button = tk.Button(text='Run Script', command=run_script)
button.pack()

app.mainloop()