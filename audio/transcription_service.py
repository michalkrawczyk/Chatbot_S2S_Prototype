"""Transcription Service - Orchestrates STT implementations"""
from general.logs import logger
from audio.stt_interface import WhisperSTT, NemoSTT, STTFactory


class TranscriptionService:
    """
    Service that orchestrates Speech-to-Text implementations
    Manages switching between different STT backends
    """
    
    def __init__(self, openai_client, default_backend="whisper"):
        """
        Initialize TranscriptionService
        
        Args:
            openai_client: Instance of OpenAIClient (for Whisper backend)
            default_backend (str): Default STT backend to use ("whisper" or "nemo")
        """
        self.openai_client = openai_client
        self.current_backend_name = None
        self.current_backend = None
        
        # Initialize with default backend
        self.switch_backend(default_backend)
    
    def switch_backend(self, backend_type):
        """
        Switch to a different STT backend
        
        Args:
            backend_type (str): Type of backend ("whisper" or "nemo")
            
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            logger.info(f"Switching STT backend to: {backend_type}")
            
            if backend_type.lower() == "whisper":
                # Create Whisper backend
                self.current_backend = WhisperSTT(self.openai_client)
                if self.current_backend.is_available():
                    self.current_backend_name = "whisper"
                    logger.info("Switched to Whisper backend successfully")
                    return True, "✓ Switched to OpenAI Whisper"
                else:
                    logger.error("Whisper backend not available")
                    return False, "❌ Whisper backend not available. Please check your API key."
            
            elif backend_type.lower() == "nemo":
                # Create Nemo backend
                self.current_backend = STTFactory.create_stt("nemo")
                if self.current_backend.is_available():
                    self.current_backend_name = "nemo"
                    logger.info("Switched to Nemo backend successfully")
                    return True, "✓ Switched to NVIDIA Nemo"
                else:
                    logger.error("Nemo backend not available")
                    return False, "❌ Nemo model failed to initialize. Please check dependencies."
            
            else:
                error_msg = f"Unsupported STT backend: {backend_type}"
                logger.error(error_msg)
                return False, f"❌ {error_msg}"
                
        except Exception as e:
            error_msg = f"Error switching STT backend: {str(e)}"
            logger.error(error_msg)
            return False, f"❌ {error_msg}"
    
    def transcribe_audio(self, audio_path, language="auto", max_retries=3):
        """
        Transcribe audio using the current STT backend
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code or "auto" for automatic detection
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Transcription text or error message
        """
        if not self.current_backend:
            logger.error("No STT backend available")
            return "No STT backend available. Please configure a backend."
        
        if not self.current_backend.is_available():
            logger.error(f"Current backend '{self.current_backend_name}' is not available")
            return f"STT backend '{self.current_backend_name}' is not available. Please check configuration."
        
        logger.info(f"Transcribing audio using {self.current_backend_name} backend")
        return self.current_backend.transcribe_audio(audio_path, language, max_retries)
    
    def get_current_backend(self):
        """
        Get the name of the current backend
        
        Returns:
            str: Name of current backend or None
        """
        return self.current_backend_name
    
    def is_available(self):
        """
        Check if the current backend is available
        
        Returns:
            bool: True if backend is available
        """
        return self.current_backend and self.current_backend.is_available()
    
    def validate_language(self, language):
        """
        Validate if language is supported by current backend
        
        Args:
            language (str): Language code to validate
            
        Returns:
            bool: True if language is supported
        """
        if not self.current_backend:
            return False
        return self.current_backend.validate_language(language)
