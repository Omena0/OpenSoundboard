from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter.messagebox as msgbox
import customtkinter as tki
from configLib import *
import soundlib as sl
import tkinter as tk
import time as t
import shutil
import sys
import os

# Set theme before window creation

# Create custom root class with DnD support
class DnDWindow(tki.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

os.makedirs('sounds/default', exist_ok=True)
os.makedirs('sounds/unused', exist_ok=True)

defaultconfig = r"""
[Devices]
output_device = "CABLE Input (VB-Audio Virtual C, MME"
input_device  = "Microphone (Logitech PRO X Gami, MME"

[Audio]
volume = 1
normalize_db = -10
min_volume = 0.05
max_volume = 3

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
root = DnDWindow()
root.title('OpenSoundBoard')
root.geometry(f'{buttonWidth * 5 + 25}x{buttonHeight * 3 + 75}')
root.wm_attributes('-topmost', True)

topBar = tki.CTkFrame(root, 500, 28)
topBar.place(x=3, y=0)

def handle_drop(event):
    # Split keeping quoted paths intact
    files = event.data.split('} {')

    for file in files:
        # Clean up path formatting
        file = file.strip('{}').strip('"')

        if file.lower().endswith('.mp3'):
            # Get filename and generate unique name
            filename = os.path.basename(file)
            dest_name = generateCopyName(f"sounds/{selectedCategory}", filename)
            dest_path = os.path.join('sounds', selectedCategory, dest_name)

            # Copy file
            shutil.copy2(file, dest_path)

    loadCategories()
    createButtons(selectedCategory)


root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', handle_drop)

def generateCopyName(path, name):
    name, extension = os.path.splitext(name)

    while os.path.exists(os.path.join(path, name) + extension):
        if name[-2].isnumeric():
            name = f'{name[:-3]}({int(name[-2]) + 1})'
        else:
            name += ' (2)'

    return name + extension

def run(*args, **kwargs):
    if callable(args[0]) and len(args) == 1 and not kwargs:
        args[0]()
        return args[0]

    def inner(callback):
        callback(*args, **kwargs)
        return callback
    return inner

def createCallback(func, *args, **kwargs):
    def callback(*args2, **kwargs2):
        args2 += args
        kwargs2.update(kwargs)
        func(*args2, **kwargs2)
    return callback

def newCategory():
    path = 'sounds'
    name = generateCopyName(path, 'New Category')

    os.makedirs(os.path.join(path, name))

    loadCategories()

def deleteCategory(category):
    if category in {'default', 'unused'}:
        msgbox.showerror('Error', 'This category cannot be deleted.')

    for sound in os.listdir(f'sounds/{category}'):
        os.rename(f'sounds/{category}/{sound}', f'sounds/unused/{sound}')
    os.rmdir(f'sounds/{category}')

    loadCategories()
    createButtons(selectedCategory)

def deleteSound(category, sound):
    os.remove(f'sounds/{category}/{sound}')
    loadCategories()
    createButtons(selectedCategory)

def renameInPlace(button, category, sound):
    sound_no_ext = sound.removesuffix(".mp3")
    button.configure(text=sound_no_ext)
    button._text_label.configure(
        wraplength=buttonWidth - 5,
        justify='center'
    )

    txt = tk.Text(
        button,
        font=button._text_label.cget("font"),
        wrap=tk.WORD,
        height=3,
        width=20
    )
    txt.tag_configure("center", justify='center')
    txt.place(
        relx=0.5, rely=0.5,
        anchor='center',
        relwidth=0.8, relheight=0.8
    )
    txt.insert("1.0", sound_no_ext)
    txt.tag_add("center", "1.0", "end")
    txt.focus_set()

    def on_click(event):
        if event.widget != txt:
            on_cancel()

    def on_confirm(event=None):
        root.unbind('<Button-1>', click_binding)
        new_name_no_ext = txt.get("1.0", "end-1c").strip()
        txt.destroy()
        button.configure(text=new_name_no_ext or sound_no_ext)
        button._text_label.configure(
            wraplength=buttonWidth - 5,
            justify='center'
        )
        if new_name_no_ext:
            old_path = f"sounds/{category}/{sound}"
            new_path = f"sounds/{category}/{new_name_no_ext}.mp3"
            os.rename(old_path, new_path)
            loadCategories()
            createButtons(category)

    def on_cancel(event=None):
        root.unbind('<Button-1>', click_binding)
        txt.destroy()
        button.configure(text=sound_no_ext)
        button._text_label.configure(
            wraplength=buttonWidth - 5,
            justify='center'
        )

    click_binding = root.bind('<Button-1>', on_click, add='+')
    txt.bind("<Return>", on_confirm)
    txt.bind("<Escape>", on_cancel)

def renameCategoryInPlace(button, category):
    button.configure(text=category)

    entry = tk.Entry(button)
    entry.place(
        relx=0.5, rely=0.5,
        anchor='center',
        relwidth=0.8, relheight=0.8
    )
    entry.insert(0, category)
    entry.focus_set()

    def on_confirm(event=None):
        new_name = entry.get().strip()
        entry.destroy()
        button.configure(text=new_name or category)
        if new_name and new_name != category:
            old_path = f"sounds/{category}"
            new_path = f"sounds/{new_name}"
            os.rename(old_path, new_path)
            loadCategories()
            createButtons(new_name)

    def on_cancel(event=None):
        entry.destroy()
        button.configure(text=category)

    entry.bind("<Return>", on_confirm)
    entry.bind("<Escape>", on_cancel)
    entry.bind("<FocusOut>", on_cancel)

def createButtons(selected_category):
    global buttons, selectedCategory
    selectedCategory = selected_category

    for button in buttons:
        button.destroy()
    buttons = []

    if selected_category not in categories:
        selected_category = 'default'

    for sound in categories[selected_category]:
        def callback(filename):
            path = os.path.join('sounds', selected_category, filename)

            # Funny callback
            def callback():
                global volume
                # Default speakers
                sl.playSound(
                    path,
                    volume=volumeSlider.get(),
                    normalize_db=config.audio.normalize_db
                )

                t.sleep(0.01)

                # VB Cable
                sl.playSound(
                    path,
                    device=config.devices.output_device,
                    volume=volumeSlider.get(),
                    normalize_db=config.audio.normalize_db
                )

            return callback

        # Create button
        b = tki.CTkButton(
            root,
            buttonWidth,
            buttonHeight,
            command=callback(sound),
            font=tki.CTkFont(
                size=min(
                    buttonHeight-36,
                    int(buttonHeight / 3.1
                        - len(sound) / 7
                        - sound.count(' ')/2
                    )
                )
            )
        )
        b.place(x=0, y=0)

        # Configure
        b.configure(text=sound.removesuffix('.mp3'))  # Set text after placement
        b._text_label.configure(wraplength=buttonWidth - 5, justify='center')

        menu = tk.Menu(
            b,
            tearoff=0,
            font=tki.CTkFont(size=14),
            takefocus=1
        )

        # "Move to" submenu
        moveToMenu = tk.Menu(
            menu,
            tearoff=0,
            font=tki.CTkFont(size=14),
            takefocus=1
        )

        # Generate "Move to" submenu options
        for category in categories:
            if category == selected_category:
                continue

            moveToMenu.add_command(
                label=category,
                command=createCallback(
                    lambda c=category, s=selected_category, f=sound:
                        (
                            os.rename(
                                f'sounds/{s}/{f}',
                                f'sounds/{c}/{f}'
                            ),
                            loadCategories(),
                            createButtons(selected_category)
                        )
                )
            )

        # Move to "default" category
        menu.add_command(
            label='Move to default',
            command=createCallback(
                lambda s=selected_category, f=sound:
                    (
                        os.rename(f'sounds/{s}/{f}', f'sounds/default/{f}'),
                        loadCategories(),
                        createButtons(selected_category)
                    )
            ),
            state='disabled' if selected_category == 'default' else 'normal'
        )

        # Rename
        menu.add_command(
            label='Rename',
            command=createCallback(
                lambda b=b, c=selected_category, s=sound:
                    renameInPlace(b, c, s)
            )
        )

        # "Move to" cascade
        menu.add_cascade(
            label='Move to',
            menu=moveToMenu
        )

        # Separator
        menu.add_separator()

        # Copy
        menu.add_command(
            label='Copy',
            command=createCallback(
                lambda s=selected_category, f=sound:
                    (
                        shutil.copy(
                            f'sounds/{s}/{f}',
                            f'sounds/{s}/{generateCopyName(f"sounds/{s}", f)}'
                        ),
                        loadCategories(),
                        createButtons(selected_category)
                    )
#               )
            )
        )

        # Delete
        menu.add_command(
            label='Delete',
            command=createCallback(
                deleteSound,
                selected_category,
                sound
            )
        )

        # Bind menu to button
        b.bind(
            '<Button-3>', createCallback(
                lambda e, menu:
                    menu.tk_popup(e.x_root, e.y_root),
                menu
            )
        )

        buttons.append(b)

    if 'configure' in globals():
        configure(root.winfo_width(), root.winfo_height(), '.')

@run
def loadCategories():
    global topButtons
    for button in topButtons:
        button.destroy()
    topButtons = []
    _menus = {}

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
            command=createCallback(createButtons, category)
        )
        b.pack(side='left', padx=2, pady=2)

        menu = tk.Menu(
            b,
            tearoff=0,
            font=tki.CTkFont(size=14),
            takefocus=1
        )
        _menus[category] = menu

        # Create new category
        menu.add_command(label='Create New', command=newCategory)

        # Rename
        menu.add_command(
            label='Rename',
            command=createCallback(
                lambda b=b, c=category:
                    renameCategoryInPlace(b, c)
            )
        )

        menu.add_separator()

        # Delete category
        if category != 'default':
            menu.add_command(
                label=f'Delete {category}',
                command=createCallback(deleteCategory, category)
            )

        def callback(e, menu):
            menu.tk_popup(e.x_root, e.y_root)

        b.bind('<Button-3>', createCallback(callback, menu))

        topButtons.append(b)

createButtons(selectedCategory)

bottomBar = tki.CTkFrame(root, 500, 28)
bottomBar.place(x=0, y=0)

# Volume %
volumeText = tki.CTkLabel(
    bottomBar,
    text='100%'
)
volumeText.place(x=8, y= - 1)


# Volume slider
def updateVolume(pos):
    volume = round(pos * 100)
    volumeText.configure(text=f'{volume}%')

# Volume slider stays at x=40
volumeSlider = tki.CTkSlider(
    bottomBar,
    width=135,
    height=20,
    from_=config.audio.min_volume,
    to=config.audio.max_volume,
    command=updateVolume
)

volumeSlider.set(config.audio.volume)
volumeSlider.place(x=40, y=3)

def resetVolume(event):
    volumeSlider.set(1.0)
    volumeText.configure(text='100%')

volumeSlider.bind("<Double-Button-1>", resetVolume)

def toggleMic():
    if micToggle.get():
        sl.startMicPassthrough(config.devices.output_device)
    else:
        sl.stopMicPassthrough()

# Mic toggle between volume and stop all
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


def configure(width, height, widget):
    global buttonsPerRow

    if str(widget) != '.':
        return

    # Reposition controls from right to left
    volumeSlider.configure(width=width * 0.8 - 175)
    micToggle.place_configure(
        x = width - stopButton.winfo_width() - micToggle.winfo_width() + 35,
        y = 3
    )
    stopButton.place_configure(
        x = width - stopButton.winfo_width() + 2,
        y = 0
    )

    bottomBar.place_configure(x=0, y=height - 33)
    bottomBar.configure(width=width)

    buttonsPerRow = width // (buttonWidth * 1.3)
    for i, button in enumerate(buttons):
        button.place_configure(
            x = 5 + (i % buttonsPerRow) * (buttonWidth * 1.3),
            y = 43 + (i // buttonsPerRow) * buttonHeight * 1.4
        )


root.bind('<Configure>', lambda e: configure(e.width, e.height, e.widget))

root.mainloop()
sl.stopAll()

sys.exit()
