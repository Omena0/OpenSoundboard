import customtkinter as tki
import soundlib as sl
import os, sys

os.makedirs('sounds/default',exist_ok=True)

# Sound config
output_device = 16 # VB CABLE INPUT
volume = 1

profiles = {}

selectedProfile = 'default'

# UI Config
buttonWidth = 66
buttonHeight = 50
buttonsPerRow = 4
buttons = []

# Create UI
root = tki.CTk()
root.title('OpenSoundBoard')
root.geometry(f'{buttonWidth*5+20}x{buttonHeight*3+75}')
root.wm_attributes('-topmost', True)


for profile in os.listdir('sounds'):
    if not os.path.isdir(f'sounds/{profile}'):
        continue

    profiles[profile] = os.listdir(f'sounds/{profile}')


def createButtons(profile):
    global buttons
    for button in buttons:
        button.place_forget()
    buttons = []
    for sound in profiles[profile]:
        def createCallback(filename):
            path = os.path.join('sounds', profile, filename)
            def callback():
                global volume
                sl.playSound(path,volume=volumeSlider.get(),normalize=True)
                sl.playSound(path,device=output_device,volume=volumeSlider.get(),normalize=True)
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

createButtons(selectedProfile)

topBar = tki.CTkFrame(root,500,28)
topBar.place(x=0,y=0)

for profile in profiles:
    def createCallback(profile):
        def callback():
            print(profile)
        return callback
    
    b = tki.CTkButton(
        topBar,
        text=profile,
        command=createCallback(profile)
    )
    b.pack(side='left')

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
    volumeText.configure(text=f'{volume}%')

# Volume slider stays at x=40
volumeSlider = tki.CTkSlider(
    bottomBar,
    width=200,
    height=20,
    from_=0.1,
    to=5,
    command=updateVolume
)
volumeSlider.place(x=40, y=3)

def toggleMic():
    if micToggle.get():
        sl.startMicPassthrough(output_device,17)
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
    width=100,  # Fixed width
    text='Stop all',  # Shorter text
    command=sl.stopAll
)
stopButton.place(x=430, y=0)

def update_positions():
    global buttonsPerRow
    width = root.winfo_width()
    height = root.winfo_height()
    # Reposition controls from right to left
    stopButton.place_configure(x=width-stopButton.winfo_width()+2, y=0)
    micToggle.place_configure(x=width-stopButton.winfo_width()-micToggle.winfo_width()+35, y=3)

    bottomBar.place_configure(x=0,y=height-30)
    bottomBar.configure(width=width)

    buttonsPerRow = width // (buttonWidth*1.3)
    for i,button in enumerate(buttons):
        button.place_configure(x=5+(i % buttonsPerRow) * (buttonWidth*1.3), y=40+(i // buttonsPerRow) * buttonHeight*1.4)


root.bind('<Configure>', lambda _: update_positions())

root.mainloop()
sl.stopAll()

sys.exit()
