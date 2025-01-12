import customtkinter as tki
import soundlib as sl
import os, sys

devicename = 16

root = tki.CTk()
root.title('OpenSoundBoard')
root.geometry('420x640')
root.wm_attributes('-topmost', True)

bottomBar = tki.CTkFrame(root,500,28)
bottomBar.place(x=0,y=0)

buttonWidth = 66
buttonHeight = 50
buttonsPerRow = 4
buttons = []

for i,file in enumerate(os.listdir('sounds')):
    def createCallback(filename):
        path = os.path.join('sounds', filename)
        def callback():
            global volume
            sl.playSound(path,volume=volumeSlider.get(),normalize=True)
            sl.playSound(path,device=devicename,volume=volumeSlider.get(),normalize=True)
        return callback

    button = tki.CTkButton(
        root,
        buttonWidth,
        buttonHeight,
        command=createCallback(file),
        font=tki.CTkFont(size=int(buttonHeight/3.5 - len(file)/10))
    )
    button.place(x=0,y=0)

    buttons.append(button)

    button.configure(text=file.removesuffix('.mp3'))  # Set text after placement
    button._text_label.configure(wraplength=buttonWidth-5, justify='center')

# Volume %
volumeText = tki.CTkLabel(
    bottomBar,
    text = '100%'
)
volumeText.place(x=0,y=0)

volume = 1

def command(pos):
    volume = round(pos*100)
    volumeText.configure(text=f'{volume}%')

# Volume slider
volumeSlider = tki.CTkSlider(
    bottomBar,
    width=100,
    height=20,
    from_=0.5,
    to=5,
    command=command
)
volumeSlider.place(x=40,y=5)

# Stop all button
stopButton = tki.CTkButton(
    bottomBar,
    width=0,
    text='Stop all sounds',
    command=sl.stopAll
)
stopButton.place(x=400,y=0)

scrollPosition = 0
def resize(event):
    global buttonsPerRow
    width = root.winfo_width()
    height = root.winfo_height()

    stopButton.place_configure(x=width-stopButton.winfo_width(),y=0)

    bottomBar.place_configure(x=0,y=height-30)
    bottomBar.configure(width=width)

    buttonsPerRow = width // (buttonWidth*1.3)
    for i,button in enumerate(buttons):
        button.place_configure(x=5+(i % buttonsPerRow) * (buttonWidth*1.3), y=scrollPosition+(i // buttonsPerRow) * buttonHeight*1.4)

root.bind('<Configure>',resize)

root.mainloop()
sl.stopAll()

sys.exit()
