import os
import time
from pathlib import Path
import tempfile

from openai import OpenAI
from general.logs import logger

SUPPORT_LANGUAGES_CAST_DICT = {
    "auto": "English",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "hi": "Hindi",
    "ko": "Korean",
    "pl": "Polish"
}

SUPPORT_LANGUAGES = list(SUPPORT_LANGUAGES_CAST_DICT.keys())

class OpenAIClient:
    """Handles OpenAI API interactions"""

    def __init__(self, default_api_key=""):
        self.client = None
        self.connected = False
        self.last_error = None
        self.default_api_key = default_api_key

        # Try to connect with default key
        if self.default_api_key:
            self.connect(self.default_api_key)

    def connect(self, api_key):
        """
        Initialize the OpenAI client with the provided API key

        Args:
            api_key (str): The OpenAI API key

        Returns:
            tuple: A tuple containing:
                - str: Status message
                - bool: Success status
                - str: Color for status display
        """
        if not api_key:
            return "Please enter your OpenAI API key", False, "red"

        try:
            # Log connection attempt without revealing the full key
            masked_key = f"***{api_key[-4:]}" if len(api_key) > 4 else "***"
            logger.info(f"Attempting to connect with API key: {masked_key}")

            client = OpenAI(api_key=api_key)
            # Test connection with a lightweight call
            client.models.list()
            self.client = client
            self.connected = True
            self.last_error = None
            logger.info("Successfully connected to OpenAI API")
            return "✓ Successfully connected to OpenAI API", True, "green"
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI connection error: {error_msg}")
            return f"❌ Error connecting to OpenAI: {error_msg}", False, "red"

    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio using OpenAI's Whisper API with retry logic

        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Transcription text or error message
        """
        if not self.client:
            return "OpenAI client not initialized. Please enter your API key."

        if not audio_path or not os.path.exists(audio_path):
            return "No audio file available for transcription."

        # Validate language
        if language and language != "auto" and not self._validate_language(language):
            return f"Unsupported language: {language}. Please select from the supported options."

        retries = 0
        while retries < max_retries:
            try:
                # Additional parameters based on language
                params = {"model": "whisper-1"}
                if language and language != "auto":
                    params["language"] = language

                logger.info(
                    f"Transcribing file: {audio_path}, language: {language}, attempt: {retries + 1}/{max_retries}")
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        **params
                    )

                logger.info(f"Transcription successful: {len(response.text)} characters")
                return response.text
            except Exception as e:
                retries += 1
                error_msg = str(e)
                logger.warning(f"Transcription attempt {retries} failed: {error_msg}")

                if retries >= max_retries:
                    logger.error(f"Transcription failed after {max_retries} attempts: {error_msg}")
                    return f"Transcription error: {error_msg}"

                # Wait before retrying with exponential backoff
                wait_time = 2 * retries
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    def _validate_language(self, language):
        """
        Validate if the language is supported by OpenAI's Whisper API

        Args:
            language (str): Language code to validate

        Returns:
            bool: True if language is supported, False otherwise
        """
        supported_languages = SUPPORT_LANGUAGES
        return language in supported_languages

    def text_to_speech(self, text, voice="alloy", stream=True):
        """
        Convert text to speech using OpenAI's TTS API with streaming support

        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use
            stream (bool): Whether to use streaming mode

        Returns:
            generator or str: Audio stream generator or file path
        """
        if not self.client:
            return "Error: Not connected to OpenAI API"

        try:
            response_format="mp3"
            # Create streaming response from OpenAI
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format=response_format,
                stream=True  # Always use streaming for efficiency
            )

            if stream:
                # Return the streaming response directly as a generator
                return response
            else:
                # Save to file for non-streaming use
                temp_dir = Path(tempfile.gettempdir()) / "spaces_audio"
                temp_dir.mkdir(exist_ok=True, parents=True)
                output_path = str(temp_dir / f"tts_{int(time.time())}.{response_format}")
                response.stream_to_file(output_path)
                return output_path

        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return f"Error generating speech: {str(e)}"
