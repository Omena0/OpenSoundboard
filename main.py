import customtkinter as tki
import soundlib as sl
import os

devicename = 'CABLE Input (VB-Audio Virtual Cable), Windows WASAPI'

root = tki.CTk()
root.title('OpenSoundBoard')
root.geometry('500x250')
root.wm_attributes('-topmost', True)

bottomBar = tki.CTkFrame(root,500,28)
bottomBar.place(x=0,y=222)

selected_files = []

for file in os.listdir('sounds'):
    def createCallback(filename):
        def callback():
            selected_files.append(os.path.join('sounds', filename))
            sl.mixSounds(selected_files, device=devicename)
        return callback

    b = tki.CTkButton(root,120,100,text=file,command=createCallback(file))
    b.pack(anchor='w')

stopAll = tki.CTkButton(bottomBar,width=100,text='Stop all sounds',command=sl.stopAll)
stopAll.place(x=400,y=0)

root.mainloop()