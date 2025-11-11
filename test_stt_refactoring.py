#!/usr/bin/env python3
"""
Simple test script to validate the STT architecture refactoring
This tests the basic structure without requiring external dependencies
"""

import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_stt_utils():
    """Test the stt_utils module"""
    print("Testing stt_utils module...")
    from stt_utils import (
        SUPPORTED_AUDIO_FORMATS, 
        SUPPORT_LANGUAGES,
        validate_audio_file,
        validate_language,
        CircuitBreaker
    )
    
    # Test constants
    assert '.wav' in SUPPORTED_AUDIO_FORMATS
    assert 'auto' in SUPPORT_LANGUAGES
    assert 'eng' in SUPPORT_LANGUAGES
    print("  ✓ Constants loaded correctly")
    
    # Test validate_language
    assert validate_language('eng')
    assert validate_language('auto')
    assert not validate_language('invalid_lang')
    print("  ✓ validate_language works correctly")
    
    # Test CircuitBreaker
    cb = CircuitBreaker(failure_threshold=3, timeout=5)
    assert not cb.is_open()
    
    cb.record_failure()
    assert not cb.is_open()
    
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()  # Should be open after 3 failures
    print("  ✓ CircuitBreaker works correctly")
    
    cb.reset()
    assert not cb.is_open()
    print("  ✓ CircuitBreaker reset works")
    
    print("✓ stt_utils module tests passed!\n")

def test_architecture():
    """Test that the architecture is properly separated"""
    print("Testing architecture separation...")
    
    # Test that OpenAIClient no longer has STT methods
    from openai_client import OpenAIClient
    
    # Create a mock client (won't actually connect without API key)
    client = OpenAIClient()
    
    # Verify STT methods are NOT present
    assert not hasattr(client, 'transcribe_audio'), "OpenAIClient should not have transcribe_audio"
    assert not hasattr(client, 'set_stt_backend'), "OpenAIClient should not have set_stt_backend"
    assert not hasattr(client, '_validate_audio_file'), "OpenAIClient should not have _validate_audio_file"
    assert not hasattr(client, '_is_circuit_open'), "OpenAIClient should not have circuit breaker"
    print("  ✓ OpenAIClient is now a pure API client")
    
    # Test that text_to_speech is still present (non-STT functionality)
    assert hasattr(client, 'text_to_speech'), "OpenAIClient should still have text_to_speech"
    assert hasattr(client, 'connect'), "OpenAIClient should still have connect"
    print("  ✓ OpenAIClient retains non-STT functionality")
    
    print("✓ Architecture separation tests passed!\n")

def test_interfaces():
    """Test STT interface structure"""
    print("Testing STT interface structure...")
    from stt_interface import STTInterface, WhisperSTT, NemoSTT
    
    # Check that abstract methods exist
    assert hasattr(STTInterface, 'transcribe_audio')
    assert hasattr(STTInterface, 'is_available')
    assert hasattr(STTInterface, 'validate_language')
    print("  ✓ STTInterface has required abstract methods")
    
    # Check that WhisperSTT has circuit breaker
    # We can't instantiate without dependencies, but we can check the class structure
    assert 'circuit_breaker' in WhisperSTT.__init__.__code__.co_names or \
           'CircuitBreaker' in str(WhisperSTT.__init__.__code__.co_consts)
    print("  ✓ WhisperSTT includes circuit breaker")
    
    # Check that NemoSTT has circuit breaker
    assert 'circuit_breaker' in NemoSTT.__init__.__code__.co_names or \
           'CircuitBreaker' in str(NemoSTT.__init__.__code__.co_consts)
    print("  ✓ NemoSTT includes circuit breaker")
    
    print("✓ STT interface tests passed!\n")

def test_transcription_service():
    """Test TranscriptionService structure"""
    print("Testing TranscriptionService structure...")
    from transcription_service import TranscriptionService
    
    # Check required methods exist
    assert hasattr(TranscriptionService, 'switch_backend')
    assert hasattr(TranscriptionService, 'transcribe_audio')
    assert hasattr(TranscriptionService, 'get_current_backend')
    assert hasattr(TranscriptionService, 'is_available')
    assert hasattr(TranscriptionService, 'validate_language')
    print("  ✓ TranscriptionService has all required methods")
    
    print("✓ TranscriptionService tests passed!\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("STT Architecture Refactoring Validation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_stt_utils()
        test_architecture()
        test_interfaces()
        test_transcription_service()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
