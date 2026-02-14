"""py2app build config for WhisperTranscribe."""

from setuptools import setup

APP = ["app.py"]
DATA_FILES = []
OPTIONS = {
    "iconfile": "icon.icns",
    "argv_emulation": False,
    "packages": ["faster_whisper", "sounddevice", "ctranslate2", "mlx_whisper", "httpx", "numpy"],
    "includes": [
        "model_manager",
        "deepgram_client",
        "nvidia_client",
        "aws_transcribe_client",
        "openai_client",
        "speechmatics_client",
        "medasr_client",
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
