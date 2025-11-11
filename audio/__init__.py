"""Audio processing and Speech-to-Text package"""

# Import structure - lazy imports to avoid dependency issues
__all__ = [
    'AudioProcessor',
    'STTInterface',
    'WhisperSTT',
    'NemoSTT',
    'STTFactory',
    'SUPPORTED_AUDIO_FORMATS',
    'SUPPORT_LANGUAGES',
    'SUPPORT_LANGUAGES_CAST_DICT',
    'validate_audio_file',
    'validate_language',
    'CircuitBreaker',
    'TranscriptionService',
]

def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies on package import"""
    if name == 'AudioProcessor':
        from audio.audio import AudioProcessor
        return AudioProcessor
    elif name in ('STTInterface', 'WhisperSTT', 'NemoSTT', 'STTFactory'):
        from audio.stt_interface import STTInterface, WhisperSTT, NemoSTT, STTFactory
        return {'STTInterface': STTInterface, 'WhisperSTT': WhisperSTT, 
                'NemoSTT': NemoSTT, 'STTFactory': STTFactory}[name]
    elif name in ('SUPPORTED_AUDIO_FORMATS', 'SUPPORT_LANGUAGES', 'SUPPORT_LANGUAGES_CAST_DICT',
                  'validate_audio_file', 'validate_language', 'CircuitBreaker'):
        from audio.stt_utils import (SUPPORTED_AUDIO_FORMATS, SUPPORT_LANGUAGES,
                                      SUPPORT_LANGUAGES_CAST_DICT, validate_audio_file,
                                      validate_language, CircuitBreaker)
        return {'SUPPORTED_AUDIO_FORMATS': SUPPORTED_AUDIO_FORMATS,
                'SUPPORT_LANGUAGES': SUPPORT_LANGUAGES,
                'SUPPORT_LANGUAGES_CAST_DICT': SUPPORT_LANGUAGES_CAST_DICT,
                'validate_audio_file': validate_audio_file,
                'validate_language': validate_language,
                'CircuitBreaker': CircuitBreaker}[name]
    elif name == 'TranscriptionService':
        from audio.transcription_service import TranscriptionService
        return TranscriptionService
    raise AttributeError(f"module 'audio' has no attribute '{name}'")
