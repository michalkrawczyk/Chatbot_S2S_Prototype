---
title: Chatbot Test V2
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
short_description: T
---

# Voice Recorder & AI Analysis

A Speech-to-Speech chatbot prototype with multiple Speech-to-Text (STT) backend options and AI-powered analysis.

## Features

- **Multiple STT Engines**: Switch between OpenAI Whisper and NVIDIA Nemo for speech-to-text transcription
- **Voice Recording**: Record audio directly from your microphone
- **Audio Upload**: Upload audio files in various formats (WAV, MP3, M4A, FLAC)
- **AI Analysis**: Analyze transcribed text using advanced AI models
- **Text-to-Speech**: Convert analysis results to speech
- **Multi-language Support**: Automatic language detection or manual selection
- **Session Memory**: Track conversation history and context

## Speech-to-Text Models

### OpenAI Whisper (Default)
- Cloud-based STT using OpenAI's Whisper API
- Requires OpenAI API key
- Excellent accuracy across multiple languages
- Fast processing

### NVIDIA Nemo
- Local STT using NVIDIA's Nemo models
- No API key required for transcription
- Can run on CPU or GPU
- Privacy-focused (audio stays on your device)

#### Installing NVIDIA Nemo Support
To use NVIDIA Nemo models (like `nvidia/parakeet-tdt-1.1b`), you need to install the `nemo_toolkit`:

```bash
pip install nemo_toolkit[asr]
```

**Note**: The default Nemo model `nvidia/parakeet-tdt-1.1b` requires `nemo_toolkit` to function properly. 
Without `nemo_toolkit`, the system will attempt to use HuggingFace transformers as a fallback, but this may not work for all Nemo models.

## Configuration

You can configure the default STT model in `general/config.py`:

```python
DEFAULT_STT_MODEL = "whisper"  # Options: "whisper", "nemo"
```

Or switch models dynamically through the UI using the "Speech-to-Text Model" dropdown.

## Usage

1. **Connect to OpenAI**: Enter your OpenAI API key (required for Whisper STT and AI analysis)
2. **Select STT Model**: Choose between Whisper or Nemo for transcription
   - For Nemo: Make sure you have installed `nemo_toolkit[asr]` first
3. **Record or Upload Audio**: Use the microphone or upload an audio file
4. **Transcribe & Analyze**: Get your transcription and AI-powered analysis
5. **Listen to Results**: Use Text-to-Speech to hear the analysis

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference