from ast import literal_eval
from typing import Any
import re
import os

_defaultconfig = ''


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


class INIConfigSection:
    def __init__(self, section: dict[str: Any]):
        self.section = section

    def __getattr__(self, key: str):
        if key in self.section:
            return self.section[key]

    def __repr__(self):
        return f'INIConfigSection({self.section})'


class INIConfig:
    def __init__(self, sections: dict[str: dict[str: Any]]):
        self._sections = sections
        self.sections = {k.lower(): INIConfigSection(v) for k, v in sections.items()}

    def __getattr__(self, key: str):
        key = key.lower()
        if key in self.sections:
            return self.sections[key]

    def __repr__(self):
        return f'<INIConfig({self._sections})>'


def setDefault(config):
    global _defaultconfig
    _defaultconfig = config


def loadConfig(filepath: str = 'config.ini', defaultconfig = None) -> INIConfig:
    """Load a ini config file from the specified file path

    Args:
        filepath (str, optional): The path of the config file. Defaults to 'config.ini'.
        defaultconfig (_type_, optional): Contents of the config file if
        it does not exist or is invalid.

        Defaults to global default config.

    Returns:
        dict: {section name: {key: value, ...}, ...}
    """

    if defaultconfig is None:
        defaultconfig = _defaultconfig

    # Create default configuration
    if not os.path.exists(filepath):
        if configPath := os.path.split(filepath)[0]:
            os.makedirs(configPath, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(defaultconfig)

    with open(filepath, 'r') as f:
        content = f.readlines()

    sections: dict[str: dict[str, Any]] = {}

    current_section = None

    for i, line in enumerate(content):
        line = line.split('#')[0].strip()

        if line.startswith('#'):
            continue

        if not line:
            continue

        # Match valid section header
        # Example: [Utils.Section_123]
        match = re.match(r'\[[A-Za-z_åäö\-\.]{1}[\wåäö\-\.]*\]', line)
        if match and match.string == line:
            current_section = line.strip('[]')
            sections[current_section] = {}
            continue

        if not current_section:
            raise INIParseError(
                f'Key definition in unspecified segment on line {line}. [{line}]'
            )

        if '=' not in line:
            raise INIParseError(f'Not a valid configuration line: {line}')

        # Parse keys
        key, value = line.split('=')
        key = key.strip()

        # Match valid keys
        match = re.match(r'[A-Za-z_\.,åäö]{1}\w*', key)
        if match and match.string != key:
            raise INIParseError(f'Invalid key: {key} on line {i}. [{line}]')

        # Parse value
        try: value = literal_eval(value)
        except Exception: value = value.strip()

        sections[current_section][key] = value

    return INIConfig(sections)


__all__ = ['loadConfig', 'INIParseError', 'setDefault']
