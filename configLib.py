from collections.abc import Any
from ast import literal_eval
import re
import os

defaultconfig = ''

class INIParseError(Exception):
    """Custom exception raised for errors encountered while parsing INI files.

    This exception is raised when the INI file being parsed has an invalid format
    or contains unexpected data.  It provides a way to handle INI parsing
    errors specifically.
    """
    def __init__(self, message="Invalid INI file format"):
        """Initializes the INIParseError with an optional error message.

        Args:
            message (str, optional): A descriptive error message. Defaults to
                "Invalid INI file format".
        """
        self.message = message
        super().__init__(self.message)

def loadConfig(filepath:str='config.ini', defaultconfig = defaultconfig) -> dict:
    """Load a ini config file from the specified file path

    Args:
        filepath (str, optional): The path of the config file. Defaults to 'config.ini'.
        defaultconfig (_type_, optional): Contents of the config file if it does not exist or is invalid.
        Defaults to global default config.

    Returns:
        dict: {section name: {key: value, ...}, ...}
    """

    # Create default configuration
    if not os.path.exists(filepath):
        os.makedirs(filepath.replace('\\','/').rsplit('/')[0],exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(defaultconfig)

    with open(filepath, 'r') as f:
        content = f.readlines()

    sections:dict[str:dict[str,Any]] = {}

    current_section = None

    for i,line in enumerate(content):
        line = line.strip()

        if line.startswith('#'):
            continue

        if re.match(r'\[[A-Za-z_åäö-.]{1}[\wåäö-.]*\]',line).string == line:
            current_section = line.strip('[]')
            sections[current_section] = {}
            continue

        if not current_section:
            raise INIParseError(f'Key definition in unspecified segment on line {line}. [{line}]')

        if '=' not in line:
            raise INIParseError(f'No = in line: {line}')

        key,value = line.split('=')
        try: value = literal_eval(value)
        except: value = value.strip()

        if re.match(r'[A-Za-z_.,åäö]{1}\w*', key).string != line:
            raise INIParseError(f'Invalid key: {key} on line {i}. [{line}]')
        




