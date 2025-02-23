import shutil
import os

build_command = """
py -m nuitka --standalone --onefile
 --product-name="OpenSoundboard" --product-version=1.0.0
 --file-description="OpenSoundboard"
 --enable-plugin=tk-inter
 --nofollow-import-to=nuitka
 --copyright="Copyright © 2025 Omena0. All rights reserved."
 --output-dir="build"
 --deployment --python-flag="-OO" --python-flag="-S"
 --output-filename="OpenSoundboard.exe"
 main.py
""".strip().replace('\n', '')


def build():
    os.makedirs('dist', exist_ok=True)
    os.system(build_command)
    shutil.move('build/OpenSoundboard.exe', 'dist/OpenSoundboard.exe')

if __name__ == "__main__":
    build()
