import os
import tempfile
import time
import uuid
from pathlib import Path

from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

from general.logs import logger


SUPPORT_LANGUAGES_CAST_DICT = {
    "auto": "English",
    "eng": "English",
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
    "pl": "Polish",
}

SUPPORT_LANGUAGES = list(SUPPORT_LANGUAGES_CAST_DICT.keys())

# Supported audio formats for validation
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}

# Non-retryable error types
NON_RETRYABLE_ERRORS = (AuthenticationError,)


class OpenAIClient:
    """Handles OpenAI API interactions"""

    def __init__(self, default_api_key="", stt_backend=None):
        self.client = None
        self.connected = False
        self.last_error = None
        self.stt_backend = stt_backend  # Optional external STT backend
        self._consecutive_failures = 0  # Circuit breaker counter
        self._circuit_open_until = 0  # Timestamp when circuit breaker opens

        # Don't store API key in memory - only use it to initialize client
        # Try to connect with default key
        if default_api_key:
            self.connect(default_api_key)

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
            client = OpenAI(api_key=api_key)
            # Test connection with a lightweight call
            client.models.list()
            self.client = client
            self.connected = True
            self.last_error = None
            self._consecutive_failures = 0  # Reset circuit breaker on success
            logger.info("Successfully connected to OpenAI API")
            return "✓ Successfully connected to OpenAI API", True, "green"
        except AuthenticationError as e:
            error_msg = "Invalid API key"
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI authentication error: {str(e)}")
            return f"❌ Authentication failed: {error_msg}", False, "red"
        except (RateLimitError, APIConnectionError) as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI connection error: {error_msg}")
            return f"❌ Error connecting to OpenAI: {error_msg}", False, "red"
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI connection error: {error_msg}")
            return f"❌ Error connecting to OpenAI: {error_msg}", False, "red"

    def _is_circuit_open(self):
        """Check if circuit breaker is open"""
        if self._circuit_open_until > time.time():
            return True
        if self._circuit_open_until > 0:
            # Circuit was open but timeout has passed, reset
            self._consecutive_failures = 0
            self._circuit_open_until = 0
        return False

    def _record_failure(self):
        """Record a failure and potentially open circuit breaker"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            # Open circuit for 60 seconds
            self._circuit_open_until = time.time() + 60
            logger.warning("Circuit breaker opened due to consecutive failures")

    def _record_success(self):
        """Record a success and reset circuit breaker"""
        self._consecutive_failures = 0
        self._circuit_open_until = 0

    def _validate_audio_file(self, audio_path):
        """
        Validate audio file format
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not audio_path or not os.path.exists(audio_path):
            return False, "No audio file available for transcription."
        
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            return False, f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        
        return True, None

    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio using configured STT backend or OpenAI's Whisper API with retry logic

        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Transcription text or error message
        """
        # Use external STT backend if configured
        if self.stt_backend:
            logger.info(f"Using external STT backend: {type(self.stt_backend).__name__}")
            return self.stt_backend.transcribe_audio(audio_path, language, max_retries)
        
        # Check circuit breaker
        if self._is_circuit_open():
            return "Service temporarily unavailable due to repeated failures. Please try again later."
        
        # Otherwise, use OpenAI Whisper (default behavior)
        if not self.client:
            return "OpenAI client not initialized. Please enter your API key."

        # Validate audio file
        is_valid, error_msg = self._validate_audio_file(audio_path)
        if not is_valid:
            return error_msg

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
                    f"Transcribing file: {audio_path}, language: {language}, attempt: {retries + 1}/{max_retries}"
                )
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file, **params
                    )

                logger.info(
                    f"Transcription successful: {len(response.text)} characters"
                )
                self._record_success()  # Reset circuit breaker on success
                return response.text
            except NON_RETRYABLE_ERRORS as e:
                # Don't retry authentication errors
                error_msg = "Authentication failed. Please check your API key."
                logger.error(f"Non-retryable error: {str(e)}")
                self._record_failure()
                return f"Transcription error: {error_msg}"
            except (RateLimitError, APIConnectionError) as e:
                retries += 1
                error_msg = str(e)
                logger.warning(f"Transcription attempt {retries} failed (retryable): {error_msg}")

                if retries >= max_retries:
                    logger.error(
                        f"Transcription failed after {max_retries} attempts: {error_msg}"
                    )
                    self._record_failure()
                    return f"Transcription error: {error_msg}"

                # Wait before retrying with exponential backoff
                wait_time = 2 * retries
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except Exception as e:
                retries += 1
                error_msg = str(e)
                logger.warning(f"Transcription attempt {retries} failed: {error_msg}")

                if retries >= max_retries:
                    logger.error(
                        f"Transcription failed after {max_retries} attempts: {error_msg}"
                    )
                    self._record_failure()
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

    def set_stt_backend(self, stt_backend):
        """
        Set the STT backend for transcription
        
        Args:
            stt_backend: Instance of STTInterface or None to use default Whisper
        """
        self.stt_backend = stt_backend
        if stt_backend:
            logger.info(f"STT backend set to: {type(stt_backend).__name__}")
        else:
            logger.info("STT backend reset to default (OpenAI Whisper)")

    def text_to_speech(self, text, voice="alloy", stream=True):
        """
        Convert text to speech using OpenAI's TTS API with streaming support

        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use
            stream (bool): Whether to use streaming mode

        Returns:
            bytes or streaming_response or str: Audio content, streaming response, or file path
        """
        if not self.client:
            return "Error: Not connected to OpenAI API"

        try:
            if stream:
                # Use the streaming response approach
                response = self.client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    instructions="Speak with soft, calm voice and a conversational manner.",
                    voice=voice,
                    input=text,
                    response_format="wav",
                )
                return response
            else:
                # Non-streaming approach for saving to file
                response = self.client.audio.speech.create(
                    model="tts-1", voice=voice, input=text
                )

                # Save to file with unique filename to avoid race conditions
                temp_dir = Path(tempfile.gettempdir()) / "spaces_audio"
                temp_dir.mkdir(exist_ok=True, parents=True)
                # Use UUID for guaranteed uniqueness instead of timestamp
                unique_id = uuid.uuid4().hex[:8]
                output_path = str(temp_dir / f"tts_{unique_id}_{int(time.time())}.mp3")
                response.stream_to_file(output_path)
                logger.info(f"Text-to-speech audio saved to: {output_path}")
                return output_path

        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return f"Error generating speech: {str(e)}"
