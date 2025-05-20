# Eye Gaze Detector

Senior design project 2024-2025 for eye gaze quadrant detection.

This application uses OpenFace to detect eye gaze vectors, segment videos into quadrants, and determine which quadrant the subject is looking at.

## Requirements

- Python 3.7+
- OpenFace
- macOS (for the installation instructions below)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NikHerdt/EyeGaze_Detector.git
cd EyeGaze_Detector
```

### 2. Set Up a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

### 4. Set Up OpenFace

Clone and set up the OpenFace repo in the openFace folder:

```bash
# Clone OpenFace repository
git clone https://github.com/TadasBaltrusaitis/OpenFace.git openFace/OpenFace
```

Follow the OpenFace wiki instructions for macOS installation:
[OpenFace Mac Installation Guide](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Mac-installation)

Quick summary of OpenFace setup:
1. Install dependencies using Homebrew
2. Configure OpenFace with CMake
3. Build OpenFace

### 5. Running the Application

Once OpenFace is set up and the Python dependencies are installed, you can run the application:

```bash
python quadrant_selector.py
```

## Usage

1. Launch the application using the command above
2. Select the input directory containing your video(s)
3. Select an output directory for the processed files
4. Set the stability threshold (default: 0.5 seconds)
5. Click "Start Processing" to begin analyzing the videos

## Troubleshooting

If you encounter git issues when pushing to the repository, try:

```bash
# If you have divergent branches, pull with rebase
git pull --rebase

# Resolve any conflicts if they occur, then push
git push

# Alternatively, merge changes
git pull
git push
```

## License

This project is licensed under the terms of the [MIT license](LICENSE).
