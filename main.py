import tkinter.messagebox as msgbox
import customtkinter as tki
from configLib import *
import soundlib as sl
import tkinter as tk
import time as t
import os, sys

os.makedirs('sounds/default',exist_ok=True)
os.makedirs('sounds/unused',exist_ok=True)

defaultconfig = r"""
[Devices]
output_device = "CABLE Input (VB-Audio Virtual Cable), Windows WASAPI"
input_device  = "Microphone Array (Realtek(R) Audio), Windows WASAPI"

[Audio]
volume = 2
normalize_db = -30
bass_boost = 10
min_volume = 0.05
max_volume = 5

"""

setDefault(defaultconfig)

config = loadConfig()

categories = {}

selectedCategory = 'default'

# UI Config
buttonWidth = 66
buttonHeight = 50
buttonsPerRow = 4
buttons = []
topButtons = []

# Create UI
root = tki.CTk()
root.title('OpenSoundBoard')
root.geometry(f'{buttonWidth*5+25}x{buttonHeight*3+75}')
root.wm_attributes('-topmost', True)

def run(*args, **kwargs):
    if callable(args[0]) and len(args) == 1 and not kwargs:
        args[0]()
        return args[0]

    def inner(callback):
        callback(*args, **kwargs)
        return callback
    return inner

def createCallback(func, *args, **kwargs):
    def callback():
        func(*args, **kwargs)
    return callback

topBar = tki.CTkFrame(root,500,28)
topBar.place(x=3,y=0)

def newCategory():
    path = 'sounds/New Category'
    while os.path.exists(path):
        if path[-2].isnumeric():
            path = f'{path[:-3]}({int(path[-2]) + 1})'
        else:
            path += ' (2)'

    os.makedirs(path)

    loadCategories()

def deleteCategory(category):
    if category in {'default','unused'}:
        msgbox.showerror('Error','This category cannot be deleted.')

    print(category)

    return

    for sound in os.listdir(f'sounds/{category}'):
        os.rename(f'sounds/{category}/{sound}',f'sounds/unused/{sound}')
    os.rmdir(f'sounds/{category}')

    loadCategories()
    createButtons(selectedCategory)

def createButtons(selected_category):
    global buttons
    for button in buttons:
        button.destroy()
    buttons = []

    if selected_category not in categories:
        selected_category = 'default'

    for sound in categories[selected_category]:
        def createCallback(filename):
            path = os.path.join('sounds', selected_category, filename)
            def callback():
                global volume
                # Default speakers
                sl.playSound(
                    path,
                    volume=volumeSlider.get()*2,
                    normalize_db=config.audio.normalize_db,
                    bass_boost=config.audio.bass_boost
                )
                
                t.sleep(0.01)
                
                # VB Cable
                sl.playSound(
                    path,
                    device=config.devices.output_device,
                    volume=volumeSlider.get()*2,
                    normalize_db=config.audio.normalize_db,
                    bass_boost=config.audio.bass_boost
                )

            return callback

        # Create button
        b = tki.CTkButton(
            root,
            buttonWidth,
            buttonHeight,
            command=createCallback(sound),
            font=tki.CTkFont(size=int(buttonHeight/3.5 - len(sound)/10))
        )
        b.place(x=0,y=0)

        b.configure(text=sound.removesuffix('.mp3'))  # Set text after placement
        b._text_label.configure(wraplength=buttonWidth-5, justify='center')

        buttons.append(b)

    if 'configure' in globals():
        configure(root.winfo_width(),root.winfo_height(),'.')

@run
def loadCategories():
    global topButtons
    for button in topButtons:
        button.destroy()
    topButtons = []

    for category in os.listdir('sounds'):
        if not os.path.isdir(f'sounds/{category}'):
            continue
        if category == 'unused':
            continue

        categories[category] = os.listdir(f'sounds/{category}')

        b = tki.CTkButton(
            topBar,
            width=0,
            height=26,
            text=category,
            command=createCallback(createButtons,category)
        )
        b.pack(side='left',padx=2,pady=2)

        menu = tk.Menu(topBar,tearoff=0)
        menu.add_command(label='Create', command=newCategory)

        if category != 'default':
            menu.add_command(label=f'Delete {category}', command=createCallback(deleteCategory,category))

        b.bind('<Button-3>', lambda e: menu.tk_popup(e.x_root,e.y_root))


        topButtons.append(b)


createButtons(selectedCategory)

bottomBar = tki.CTkFrame(root,500,28)
bottomBar.place(x=0,y=0)

# Volume %
volumeText = tki.CTkLabel(
    bottomBar,
    text = '100%'
)
volumeText.place(x=8,y=-1)


# Volume slider
def updateVolume(pos):
    volume = round(pos*100)
    volumeText.configure(text=f'{volume//2}%')

# Volume slider stays at x=40
volumeSlider = tki.CTkSlider(
    bottomBar,
    width=135,
    height=20,
    from_=config.audio.min_volume*2,
    to=config.audio.max_volume*2,
    command=updateVolume
)

volumeSlider.set(config.audio.volume)
volumeSlider.place(x=40, y=3)

def toggleMic():
    if micToggle.get():
        sl.startMicPassthrough(config.devices.output_device,config.devices.input_device)
    else:
        sl.stopMicPassthrough()

# Mic toggle between volume and stop
micToggle = tki.CTkSwitch(
    bottomBar,
    text="Mic",  # Shorter text to fit
    command=toggleMic
)
micToggle.place(x=0, y=0)  # Right after volume slider

# Stop button after mic toggle
stopButton = tki.CTkButton(
    bottomBar,
    width=0,  # Fixed width
    text='Stop all',  # Shorter text
    command=sl.stopAll
)
stopButton.place(x=430, y=0)

def configure(width,height,widget):
    global buttonsPerRow

    if str(widget) != '.':
        return

    # Reposition controls from right to left
    volumeSlider.configure(width=width*0.8-175)
    micToggle.place_configure(x=width-stopButton.winfo_width()-micToggle.winfo_width()+35, y=3)
    stopButton.place_configure(x=width-stopButton.winfo_width()+2, y=0)

    bottomBar.place_configure(x=0,y=height-33)
    bottomBar.configure(width=width)

    buttonsPerRow = width // (buttonWidth*1.3)
    for i,button in enumerate(buttons):
        button.place_configure(x=5+(i % buttonsPerRow) * (buttonWidth*1.3), y=43+(i // buttonsPerRow) * buttonHeight*1.4)

root.bind('<Configure>', lambda e: configure(e.width,e.height,e.widget))


root.mainloop()
sl.stopAll()

sys.exit()
