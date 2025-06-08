#!/usr/bin/env python3
"""
Setup script for the AI Music Composition Web App
This script creates the necessary directory structure and extracts the original code.
"""

import os
import sys
import shutil
from pathlib import Path

# Define project structure
PROJECT_STRUCTURE = {
    "templates": ["index.html"],
    "static": {
        "css": [],
        "js": [],
        "output": []
    },
    "compositions": []
}

def create_directory_structure():
    """Create the project directory structure"""
    current_dir = Path.cwd()
    
    print(f"Creating project structure in {current_dir}")
    
    def create_dirs(structure, parent_path=current_dir):
        for key, value in structure.items() if isinstance(structure, dict) else []:
            path = parent_path / key
            path.mkdir(exist_ok=True)
            print(f"Created directory: {path}")
            
            if isinstance(value, dict):
                create_dirs(value, path)
            elif isinstance(value, list):
                # These would be files, but we don't create them here
                pass
    
    create_dirs(PROJECT_STRUCTURE)

def extract_original_code():
    """Extract the original code from paste.txt"""
    source_file = Path("paste.txt")
    if not source_file.exists():
        print("Error: paste.txt not found!")
        return False
    
    # Copy the original file
    shutil.copy(source_file, "paste_original.py")
    print(f"Copied {source_file} to paste_original.py")
    
    return True

def generate_soundfont_info():
    """Create a text file with information about obtaining SoundFonts"""
    soundfont_info = """
# SoundFont Information

To convert MIDI to audio, this application requires a SoundFont (.sf2) file.
SoundFonts provide the instrument sounds needed for MIDI playback.

## Recommended SoundFonts

1. FluidR3_GM.sf2 - A high-quality, general MIDI SoundFont
   - Size: ~140MB
   - Download: https://musical-artifacts.com/artifacts/609

2. GeneralUser GS v1.471.sf2 - Another popular general MIDI SoundFont
   - Size: ~30MB
   - Download: https://musical-artifacts.com/artifacts/523

## Installation

1. Download one of the SoundFonts above
2. Place the .sf2 file in a location accessible to the application
3. Set the path in your .env file:
   ```
   SOUNDFONT_PATH=/path/to/your/soundfont.sf2
   ```

## Using Without a SoundFont

If you don't have a SoundFont file, the application will create dummy audio files.
These files won't contain actual audio but will allow testing the application flow.
"""
    
    with open("soundfont_info.md", "w") as f:
        f.write(soundfont_info)
    
    print("Created soundfont_info.md with information about obtaining SoundFonts")

def main():
    """Run the setup process"""
    print("Setting up AI Music Composition Web App")
    print("======================================")
    
    create_directory_structure()
    
    if extract_original_code():
        print("\nOriginal code extracted successfully!")
    else:
        print("\nWarning: Could not extract original code. Make sure paste.txt exists.")
    
    generate_soundfont_info()
    
    print("\nSetup complete! Next steps:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Download a SoundFont file (see soundfont_info.md)")
    print("3. Create a .env file with your API keys (copy from .env.example)")
    print("4. Run the application: python run.py")

if __name__ == "__main__":
    main()