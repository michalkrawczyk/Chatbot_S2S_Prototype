"""Speech-to-Text Interface and Implementations"""
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from general.logs import logger

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Supported audio formats for validation
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}


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
            openai_client: Instance of OpenAIClient
        """
        self.openai_client = openai_client
        logger.info("WhisperSTT initialized")

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
        # Delegate to the OpenAI client's existing implementation
        return self.openai_client.transcribe_audio(audio_path, language, max_retries)

    def is_available(self):
        """
        Check if Whisper service is available
        
        Returns:
            bool: True if OpenAI client is connected
        """
        return self.openai_client.connected if self.openai_client else False

    def validate_language(self, language):
        """
        Validate if the language is supported by Whisper
        
        Args:
            language (str): Language code to validate
            
        Returns:
            bool: True if language is supported
        """
        return self.openai_client._validate_language(language) if self.openai_client else False


class NemoSTT(STTInterface):
    """NVIDIA Nemo implementation of STT"""

    def __init__(self, model_name="nvidia/parakeet-tdt-1.1b", target_sample_rate=16000):
        """
        Initialize NemoSTT with specified model
        
        Args:
            model_name (str): Name of the Nemo model to use
            target_sample_rate (int): Target sample rate for audio processing (default: 16000 Hz)
        """
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.model = None
        self.processor = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Nemo model"""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch
            
            logger.info(f"Loading Nemo model: {self.model_name}")
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self.model.to(device)
            self.device = device
            
            logger.info(f"NemoSTT initialized successfully with model: {self.model_name}, target sample rate: {self.target_sample_rate} Hz")
        except Exception as e:
            logger.error(f"Error initializing Nemo model: {str(e)}")
            self.model = None
            self.processor = None

    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio using NVIDIA Nemo
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection (not used for Nemo)
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Transcription text or error message
        """
        if not self.model or not self.processor:
            return "Nemo model not initialized. Please check installation."
        
        if not audio_path or not os.path.exists(audio_path):
            return "No audio file available for transcription."
        
        # Validate audio file format
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            return f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        
        retries = 0
        while retries < max_retries:
            try:
                if not TORCH_AVAILABLE:
                    return "Torch and torchaudio libraries not available. Please install them to use Nemo."
                
                logger.info(f"Transcribing with Nemo: {audio_path}, attempt: {retries + 1}/{max_retries}")
                
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                    waveform = resampler(waveform)
                    sample_rate = self.target_sample_rate
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Prepare inputs
                inputs = self.processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate transcription
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs)
                
                # Decode transcription
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                logger.info(f"Nemo transcription successful: {len(transcription)} characters")
                return transcription
                
            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except RuntimeError as e:
                # Runtime errors (like CUDA OOM) might be retryable
                retries += 1
                error_msg = str(e)
                logger.warning(f"Nemo transcription attempt {retries} failed (retryable): {error_msg}")
                
                if retries >= max_retries:
                    logger.error(f"Nemo transcription failed after {max_retries} attempts: {error_msg}")
                    return f"Transcription error: {error_msg}"
                
                # Wait before retrying
                wait_time = 2 * retries
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            except Exception as e:
                # Other errors are likely non-retryable (like invalid file format after we already checked)
                error_msg = str(e)
                logger.error(f"Nemo transcription failed: {error_msg}")
                return f"Transcription error: {error_msg}"

    def is_available(self):
        """
        Check if Nemo service is available
        
        Returns:
            bool: True if model is loaded
        """
        return self.model is not None and self.processor is not None

    def validate_language(self, language):
        """
        Validate if the language is supported by Nemo
        
        Note: Nemo models may have different language support depending on the model.
        This implementation assumes multilingual support.
        
        Args:
            language (str): Language code to validate
            
        Returns:
            bool: True (assumes multilingual support)
        """
        # Nemo models typically support multiple languages
        # For simplicity, we return True, but this could be model-specific
        return True


class STTFactory:
    """Factory class to create STT instances"""
    
    @staticmethod
    def create_stt(model_type, **kwargs):
        """
        Create an STT instance based on model type
        
        Args:
            model_type (str): Type of STT model ("whisper" or "nemo")
            **kwargs: Additional arguments for STT initialization
            
        Returns:
            STTInterface: Instance of the requested STT implementation
        """
        if model_type.lower() == "whisper":
            openai_client = kwargs.get("openai_client")
            if not openai_client:
                raise ValueError("openai_client is required for WhisperSTT")
            return WhisperSTT(openai_client)
        elif model_type.lower() == "nemo":
            model_name = kwargs.get("model_name", "nvidia/parakeet-tdt-1.1b")
            target_sample_rate = kwargs.get("target_sample_rate", 16000)
            return NemoSTT(model_name, target_sample_rate)
        else:
            raise ValueError(f"Unsupported STT model type: {model_type}")
