import tkinter as tk
import tkinter.ttk as ttk
import subprocess
import pystray
from PIL import Image, ImageDraw

def run_script():
    subprocess.run(['python', 'src/emotions.py'])

def create_menu(icon):
    menu = pystray.Menu(pystray.MenuItem('Run Script', on_click=run_script))
    icon.menu = menu

app = tk.Tk()
app.withdraw()

image = Image.new('RGB', (24, 24), 'white')
draw = ImageDraw.Draw(image)
draw.text((2, 2), 'My App', fill='black')

icon = pystray.Icon('My App', image, menu=create_menu)
icon.run()
