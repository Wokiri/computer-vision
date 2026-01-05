# ImageLab - Image Processing Application

## Project Overview

ImageLab is a comprehensive desktop application built with PyQt5 for image processing and computer vision tasks. It provides an intuitive graphical interface for performing various operations on images including resizing, filtering, and object detection.

## Features

- **Image Upload & Display**: Support for multiple image formats (PNG, JPG, JPEG, BMP, TIFF)
- **Smart Resizing**: 
  - Maintain aspect ratio
  - Content-aware resizing (seam carving)
  - Precise dimension control
- **Advanced Filters**:
  - Grayscale conversion
  - Blur and sharpening
  - Edge detection
  - Sepia tone
  - Brightness and contrast adjustment
- **Object Detection**: Basic face detection using Haar cascades
- **Zoom Controls**: Standard zoom levels and custom zoom in/out
- **Real-time Preview**: Instant visualization of processing results

## Installation

### Recommended: Virtual Environment

Use a virtual environment to manage dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment

# On Windows:
.\.venv\Scripts\activate

# On macOS/Linux:
source ./.venv/bin/activate


# Install Dependencies from requirements.txt
python -m pip install -r requirements.txt


# Verify Installation
pyuic5 --version

```

## To run the program:

### If running on linux distros:


```bash

# Make the script executable:
chmod +x convert-ui.sh

# With default path
./convert-ui.sh

# With custom path
./convert-ui.sh --uipath /path/to/your/uidesigns

# Show help
./convert-ui.sh --help

```

### If windows:

```bash

# Navigate to script directory then run:
.\convert-ui.ps1

# With parameters:
.\convert-ui.ps1 -UiPath "C:\my\uidesigns"
.\convert-ui.ps1 -Help
.\convert-ui.ps1 -UiPath "C:\my\uidesigns" -Help

```


### Then


```bash

python image_lab.py

```
