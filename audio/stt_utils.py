"""Shared utilities for Speech-to-Text implementations"""
import os
import time
from general.logs import logger


# Supported audio formats for validation
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}

# Supported languages for STT
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


def validate_audio_file(audio_path):
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


def validate_language(language, supported_languages=None):
    """
    Validate if the language is supported
    
    Args:
        language (str): Language code to validate
        supported_languages (list, optional): List of supported languages. 
                                             Defaults to SUPPORT_LANGUAGES
        
    Returns:
        bool: True if language is supported, False otherwise
    """
    if supported_languages is None:
        supported_languages = SUPPORT_LANGUAGES
    return language in supported_languages


def retry_with_backoff(func, max_retries=3, retryable_exceptions=None, non_retryable_exceptions=None):
    """
    Execute a function with retry logic and exponential backoff
    
    Args:
        func (callable): Function to execute (should take no arguments)
        max_retries (int): Maximum number of retry attempts
        retryable_exceptions (tuple): Tuple of exception types to retry on
        non_retryable_exceptions (tuple): Tuple of exception types not to retry on
        
    Returns:
        The return value of func if successful
        
    Raises:
        The last exception if all retries fail
    """
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            # Check if it's a non-retryable error
            if non_retryable_exceptions and isinstance(e, non_retryable_exceptions):
                logger.error(f"Non-retryable error: {str(e)}")
                raise
            
            # Check if it's a retryable error
            if retryable_exceptions and not isinstance(e, retryable_exceptions):
                # If retryable_exceptions is specified and this isn't one, don't retry
                logger.error(f"Non-retryable error type: {str(e)}")
                raise
            
            retries += 1
            last_exception = e
            logger.warning(f"Attempt {retries} failed: {str(e)}")
            
            if retries >= max_retries:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            
            # Wait before retrying with exponential backoff
            wait_time = 2 * retries
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures
    """
    
    def __init__(self, failure_threshold=5, timeout=60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold (int): Number of consecutive failures before opening circuit
            timeout (int): Timeout in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self._consecutive_failures = 0
        self._circuit_open_until = 0
    
    def is_open(self):
        """
        Check if circuit breaker is open
        
        Returns:
            bool: True if circuit is open (requests should be blocked)
        """
        if self._circuit_open_until > time.time():
            return True
        if self._circuit_open_until > 0:
            # Circuit was open but timeout has passed, reset
            self._consecutive_failures = 0
            self._circuit_open_until = 0
        return False
    
    def record_failure(self):
        """Record a failure and potentially open circuit breaker"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            # Open circuit for timeout seconds
            self._circuit_open_until = time.time() + self.timeout
            logger.warning(f"Circuit breaker opened due to {self._consecutive_failures} consecutive failures")
    
    def record_success(self):
        """Record a success and reset circuit breaker"""
        self._consecutive_failures = 0
        self._circuit_open_until = 0
    
    def reset(self):
        """Manually reset the circuit breaker"""
        self._consecutive_failures = 0
        self._circuit_open_until = 0
