# Orpheus-TTS-Local (OpenAI API Edition)
## All the credit goes to [isaiahbjork](https://github.com/isaiahbjork/orpheus-tts-local) I only modified the gguf_orpheus.py with Gemini 2.0 Flash 01-21

A lightweight client for running [Orpheus TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) locally using LM Studio API.

## Features

- 🎧 High-quality Text-to-Speech using the Orpheus TTS model
- 💻 Completely local - no cloud API keys needed
- 🔊 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe)
- 💾 Save audio to WAV files

## Quick Setup

1. Install [LM Studio](https://lmstudio.ai/) 
2. Download the [Orpheus TTS model (orpheus-3b-0.1-ft-q4_k_m.gguf)](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) in LM Studio
3. Load the Orpheus model in LM Studio
4. Start the local server in LM Studio (default: http://127.0.0.1:1234)
5. Clone this repo ```git clone https://github.com/AlgorithmicKing/orpheus-tts-local-openai.git```
6. Install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
7. Run the script:
   ```
   python gguf_orpheus.py
   ```

### TTS Settings in OpenWebUI

- Text-to-Speech Engine: OpenAI.
- URL: http://localhost:5000/v1.
- API: not-needed.
- TTS Voice: Any from the "Available Voices" section.
- TTS Model: not-needed.

### Options

- `--list-voices`: List available TTS voices and exit. This will print the list of available voices to the console and then stop the server from starting.
- `--host`: The host address to bind the API server to. (default: `0.0.0.0` - meaning it will listen on all available network interfaces, making it accessible from other machines on your network).
- `--port`:  The port number to run the API server on. (default: `5000`).
- `--debug`: Run the Flask API server in debug mode. Useful for development and seeing detailed error messages, but not recommended for production.

## Available Voices

- tara - Best overall voice for general use (default)
- leah
- jess
- leo
- dan
- mia
- zac
- zoe

## Emotion
You can add emotion to the speech by adding the following tags:
```xml
<giggle>
<laugh>
<chuckle>
<sigh>
<cough>
<sniffle>
<groan>
<yawn>
<gasp>
```

## License

Apache 2.0

