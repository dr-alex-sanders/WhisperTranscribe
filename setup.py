"""py2app build config for WhisperTranscribe."""

import sys
sys.setrecursionlimit(5000)

from setuptools import setup

APP = ["app.py"]
DATA_FILES = []
OPTIONS = {
    "iconfile": "icon.icns",
    "argv_emulation": False,
    "frameworks": [
        "/Users/alexsanders/Documents/Vosk/.venv/lib/python3.14/site-packages/_sounddevice_data/portaudio-binaries/libportaudio.dylib",
    ],
    "packages": ["sounddevice", "_sounddevice_data", "mlx_whisper", "httpx", "numpy"],
    "includes": [
        "model_manager",
    ],
    "excludes": [
        "torch",
        "transformers",
        "boto3",
        "botocore",
        "resemblyzer",
        "sklearn",
        "scipy",
        "pytest",
        "pip",
        "setuptools",
        "onnxruntime",
        "numba",
        "llvmlite",
        "sympy",
        "pygments",
        "PIL",
        "Pillow",
        "matplotlib",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "jedi",
        "parso",
        "docutils",
        "sphinx",
    ],
    "plist": {
        "CFBundleName": "WhisperTranscribe",
        "CFBundleDisplayName": "WhisperTranscribe",
        "CFBundleIdentifier": "com.whispertranscribe.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "NSMicrophoneUsageDescription": "WhisperTranscribe needs microphone access for speech-to-text.",
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
