"""Speech-to-Text Interface and Implementations"""
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

from general.logs import logger
from audio.stt_utils import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORT_LANGUAGES,
    validate_audio_file,
    validate_language,
    CircuitBreaker,
)

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Non-retryable error types for Whisper
NON_RETRYABLE_ERRORS = (AuthenticationError,)


class STTInterface(ABC):
    """Abstract base class for Speech-to-Text implementations"""

    @abstractmethod
    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio file to text
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Transcription text or error message
        """
        pass

    @abstractmethod
    def is_available(self):
        """
        Check if the STT service is available/initialized
        
        Returns:
            bool: True if service is available, False otherwise
        """
        pass

    @abstractmethod
    def validate_language(self, language):
        """
        Validate if the language is supported
        
        Args:
            language (str): Language code to validate
            
        Returns:
            bool: True if language is supported, False otherwise
        """
        pass


class WhisperSTT(STTInterface):
    """OpenAI Whisper implementation of STT"""

    def __init__(self, openai_client):
        """
        Initialize WhisperSTT with OpenAI client
        
        Args:
            openai_client: Instance of OpenAIClient (used only for accessing the OpenAI API client)
        """
        self.openai_client = openai_client
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        logger.info("WhisperSTT initialized with circuit breaker")

    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio using OpenAI's Whisper API
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Transcription text or error message
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            return "Service temporarily unavailable due to repeated failures. Please try again later."
        
        # Check if client is initialized
        if not self.openai_client or not self.openai_client.client:
            return "OpenAI client not initialized. Please enter your API key."

        # Validate audio file
        is_valid, error_msg = validate_audio_file(audio_path)
        if not is_valid:
            return error_msg

        # Validate language
        if language and language != "auto" and not self.validate_language(language):
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
                    response = self.openai_client.client.audio.transcriptions.create(
                        file=audio_file, **params
                    )

                logger.info(
                    f"Transcription successful: {len(response.text)} characters"
                )
                self.circuit_breaker.record_success()
                return response.text
            except NON_RETRYABLE_ERRORS as e:
                error_msg = "Authentication failed. Please check your API key."
                logger.error(f"Non-retryable error: {str(e)}")
                self.circuit_breaker.record_failure()
                return f"Transcription error: {error_msg}"
            except (RateLimitError, APIConnectionError) as e:
                retries += 1
                error_msg = str(e)
                logger.warning(f"Transcription attempt {retries} failed (retryable): {error_msg}")

                if retries >= max_retries:
                    logger.error(
                        f"Transcription failed after {max_retries} attempts: {error_msg}"
                    )
                    self.circuit_breaker.record_failure()
                    return f"Transcription error: {error_msg}"

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
                    self.circuit_breaker.record_failure()
                    return f"Transcription error: {error_msg}"

                wait_time = 2 * retries
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    def is_available(self):
        """
        Check if Whisper service is available

        Returns:
            bool: True if OpenAI client is initialized
        """
        return self.openai_client is not None

    def validate_language(self, language):
        """
        Validate if the language is supported by Whisper

        Args:
            language (str): Language code to validate

        Returns:
            bool: True if language is supported
        """
        return validate_language(language, SUPPORT_LANGUAGES)


class NemoSTT(STTInterface):
    """NVIDIA Canary implementation of STT"""

    def __init__(self, model_name="nvidia/canary-1b", target_sample_rate=16000):
        """
        Initialize CanarySTT with Canary model

        Args:
            model_name (str): Name of the Canary model to use (default: nvidia/canary-1b)
            target_sample_rate (int): Target sample rate for audio processing (default: 16000 Hz)
        """
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.model = None
        self.processor = None
        self.device = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Canary model using nemo_toolkit"""
        try:
            import nemo.collections.asr as nemo_asr

            logger.info(f"Loading Canary model: {self.model_name}")
            start_time = time.time()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            self.device = device

            if device == "cuda":
                self.model = self.model.to(device)
                try:
                    self.model = self.model.half()
                    logger.info("Enabled half-precision (FP16) for faster inference")
                except Exception as e:
                    logger.warning(f"Could not enable half-precision: {e}")

            self.model.eval()
            self.processor = None

            load_time = time.time() - start_time
            logger.info(f"Canary STT initialized successfully: {self.model_name} (loaded in {load_time:.2f}s)")

        except ImportError as e:
            logger.error(f"Error importing nemo_toolkit: {str(e)}")
            logger.error("To use NVIDIA Canary model, install: pip install nemo_toolkit[asr]")
            self.model = None
            self.processor = None
        except Exception as e:
            logger.error(f"Error initializing Canary model: {str(e)}")
            self.model = None
            self.processor = None

    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio using NVIDIA Canary model

        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Transcription text or error message
        """
        if self.circuit_breaker.is_open():
            return "Service temporarily unavailable due to repeated failures. Please try again later."

        if not self.model:
            return "Canary model not initialized. Please install nemo_toolkit[asr]."

        is_valid, error_msg = validate_audio_file(audio_path)
        if not is_valid:
            return error_msg

        if language and language != "auto" and not self.validate_language(language):
            return f"Unsupported language: {language}. Please select from the supported options."

        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Transcribing with Canary: {audio_path}, language: {language}, attempt: {retries + 1}/{max_retries}")

                # Prepare Canary prompt for language-specific transcription
                if language and language != "auto":
                    # Convert "eng" to "en", keep others as-is
                    lang_code = "en" if language == "eng" else language
                    # Canary prompt format: "<|source_lang|><|transcribe|>"
                    prompt = f"<|{lang_code}|><|transcribe|>"
                    logger.info(f"Using Canary prompt: {prompt}")
                    hypotheses = self.model.transcribe(
                        [audio_path],
                        batch_size=1,
                        prompt=prompt
                    )
                else:
                    # Auto-detect language
                    logger.info("Using auto language detection")
                    hypotheses = self.model.transcribe([audio_path], batch_size=1)

                if hypotheses and len(hypotheses) > 0:
                    transcription = hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
                else:
                    transcription = ""

                logger.info(f"Canary transcription successful: {len(transcription)} characters")
                self.circuit_breaker.record_success()
                return transcription

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except RuntimeError as e:
                retries += 1
                error_msg = str(e)
                logger.warning(f"Canary transcription attempt {retries} failed (retryable): {error_msg}")

                if retries >= max_retries:
                    logger.error(f"Canary transcription failed after {max_retries} attempts: {error_msg}")
                    self.circuit_breaker.record_failure()
                    return f"Transcription error: {error_msg}"

                wait_time = 2 * retries
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Canary transcription failed: {error_msg}")
                self.circuit_breaker.record_failure()
                return f"Transcription error: {error_msg}"

    def is_available(self):
        """
        Check if Canary service is available

        Returns:
            bool: True if model is loaded successfully
        """
        return self.model is not None

    def validate_language(self, language):
        """
        Validate if the language is supported by Canary

        Args:
            language (str): Language code to validate

        Returns:
            bool: True if language is supported by Canary
        """
        # Canary-1b supported languages
        canary_supported_languages = [
            "auto", "eng", "es", "fr", "de", "it", "pt", "nl",
            "ru", "zh", "ja", "ar", "hi", "ko", "pl"
        ]
        return validate_language(language, canary_supported_languages)


class STTFactory:
    """Factory class to create STT instances"""

    @staticmethod
    def create_stt(model_type, **kwargs):
        """
        Create an STT instance based on model type

        Args:
            model_type (str): Type of STT model ("whisper" or "canary")
            **kwargs: Additional arguments for STT initialization

        Returns:
            STTInterface: Instance of the requested STT implementation
        """
        if model_type.lower() == "whisper":
            openai_client = kwargs.get("openai_client")
            if not openai_client:
                raise ValueError("openai_client is required for WhisperSTT")
            return WhisperSTT(openai_client)
        elif model_type.lower() in ["nemo", "canary"]:
            model_name = kwargs.get("model_name", "nvidia/canary-1b")
            target_sample_rate = kwargs.get("target_sample_rate", 16000)
            return NemoSTT(model_name, target_sample_rate)
        else:
            raise ValueError(f"Unsupported STT model type: {model_type}. Supported: 'whisper', 'canary'")