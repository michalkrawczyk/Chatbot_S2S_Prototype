"""Tests for audio/ modules: stt_utils, transcription_service, stt_interface."""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. test_circuit_breaker_lifecycle
# ---------------------------------------------------------------------------

def test_circuit_breaker_lifecycle():
    from audio.stt_utils import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, timeout=60)

    # 1. Fresh breaker is closed.
    assert not cb.is_open()

    # 2. Record failure_threshold - 1 failures → still closed.
    for _ in range(cb.failure_threshold - 1):
        cb.record_failure()
    assert not cb.is_open()

    # 3. One more failure hits threshold → open.
    cb.record_failure()
    assert cb.is_open()

    # 4. record_success() while open → still open (open_until time hasn't elapsed).
    cb.record_success()
    assert cb.is_open()

    # 5. reset() → closed.
    cb.reset()
    assert not cb.is_open()

    # 6. record_success on clean breaker → failure count stays at 0.
    cb.record_success()
    assert cb._consecutive_failures == 0


# ---------------------------------------------------------------------------
# 2. test_audio_validation_pipeline
# ---------------------------------------------------------------------------

def test_audio_validation_pipeline(tmp_path):
    from audio.stt_utils import validate_audio_file, validate_language

    # 1. None path → (False, error_msg).
    ok, msg = validate_audio_file(None)
    assert not ok
    assert msg

    # 2. Non-existent path → (False, error_msg).
    ok, msg = validate_audio_file("/nonexistent/path.wav")
    assert not ok
    assert msg

    # 3. Real empty tempfile with .mp3 extension → (True, None).
    mp3 = tmp_path / "test.mp3"
    mp3.touch()
    ok, msg = validate_audio_file(str(mp3))
    assert ok
    assert msg is None

    # 4. Rename to .exe → (False, error_msg).
    exe = tmp_path / "test.exe"
    mp3.rename(exe)
    ok, msg = validate_audio_file(str(exe))
    assert not ok
    assert msg

    # 5. "eng" with default list → True.
    assert validate_language("eng") is True

    # 6. "eng" with custom list ["de", "fr"] → False.
    assert validate_language("eng", ["de", "fr"]) is False

    # 7. "xyz" → False.
    assert validate_language("xyz") is False


# ---------------------------------------------------------------------------
# 3. test_retry_with_backoff_success_and_failure
# ---------------------------------------------------------------------------

def test_retry_with_backoff_success_and_failure():
    from audio.stt_utils import retry_with_backoff

    # 1. Succeeds on first try → returns value, sleep not called.
    with patch("audio.stt_utils.time") as mock_time:
        func = MagicMock(return_value="result")
        result = retry_with_backoff(func, max_retries=3)
        assert result == "result"
        mock_time.sleep.assert_not_called()

    # 2. Fails once then succeeds → returns value, sleep called once.
    with patch("audio.stt_utils.time") as mock_time:
        func = MagicMock(side_effect=[Exception("fail"), "result"])
        result = retry_with_backoff(func, max_retries=3)
        assert result == "result"
        assert mock_time.sleep.call_count == 1

    # 3. Always raises retryable exception → raises after max_retries.
    with patch("audio.stt_utils.time"):
        func = MagicMock(side_effect=Exception("always fails"))
        with pytest.raises(Exception, match="always fails"):
            retry_with_backoff(func, max_retries=3)

    # 4. Raises non-retryable exception → raises immediately, no retries.
    class _NonRetryable(Exception):
        pass

    with patch("audio.stt_utils.time") as mock_time:
        func = MagicMock(side_effect=_NonRetryable("bad"))
        with pytest.raises(_NonRetryable):
            retry_with_backoff(
                func,
                max_retries=3,
                non_retryable_exceptions=(_NonRetryable,),
            )
        assert func.call_count == 1
        mock_time.sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 11. test_transcription_service_backend_switching
# ---------------------------------------------------------------------------

def test_transcription_service_backend_switching():
    from audio.stt_interface import STTFactory
    from audio.transcription_service import TranscriptionService

    mock_client = MagicMock()
    mock_client.client = MagicMock()

    # 1. Default "whisper" backend → current backend name and availability.
    svc = TranscriptionService(mock_client, default_backend="whisper")
    assert svc.get_current_backend() == "whisper"
    assert svc.is_available() is True

    # 2. switch_backend("whisper") already set → (True, msg).
    ok, msg = svc.switch_backend("whisper")
    assert ok is True
    assert msg

    # 3. switch_backend("nemo") with unavailable mock → (False, msg); backend name unchanged.
    with patch.object(STTFactory, "create_stt") as mock_factory:
        mock_nemo = MagicMock()
        mock_nemo.is_available.return_value = False
        mock_factory.return_value = mock_nemo
        ok, msg = svc.switch_backend("nemo")
        assert ok is False
        assert msg
    assert svc.get_current_backend() == "whisper"

    # Restore a working whisper backend for subsequent steps.
    svc.switch_backend("whisper")

    # 4. switch_backend("invalid") → (False, msg).
    ok, msg = svc.switch_backend("invalid")
    assert ok is False
    assert msg

    # 5. validate_language("eng") delegated to current backend → True.
    assert svc.validate_language("eng") is True

    # 6. validate_language("xyz") → False.
    assert svc.validate_language("xyz") is False

    # 7. current_backend.is_available() mocked False → transcribe returns error string.
    svc.current_backend.is_available = MagicMock(return_value=False)
    result = svc.transcribe_audio("path.wav")
    assert isinstance(result, str)
    assert result


# ---------------------------------------------------------------------------
# 12. test_whisper_stt_validation_and_circuit_breaker
# ---------------------------------------------------------------------------

def test_whisper_stt_validation_and_circuit_breaker(tmp_path):
    from audio.stt_interface import WhisperSTT

    stub_client = MagicMock()
    stub_client.client = MagicMock()
    whisper = WhisperSTT(openai_client=stub_client)

    # 1. is_available() → True (openai_client is not None).
    assert whisper.is_available() is True

    # 2. transcribe_audio(None) → audio validation error string.
    result = whisper.transcribe_audio(None)
    assert isinstance(result, str)
    assert result

    # 3. transcribe_audio("/nonexistent.wav") → audio validation error string.
    result = whisper.transcribe_audio("/nonexistent.wav")
    assert isinstance(result, str)
    assert result

    # 4. Valid .wav file with unsupported language → language error string.
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF fake wav")
    result = whisper.transcribe_audio(str(wav), language="xyz")
    assert isinstance(result, str)
    assert "language" in result.lower() or "unsupported" in result.lower()

    # 5. Trigger circuit breaker: 5 failures → is_open() == True.
    for _ in range(5):
        whisper.circuit_breaker.record_failure()
    assert whisper.circuit_breaker.is_open() is True

    # 6. transcribe_audio while circuit open → "temporarily unavailable" string.
    result = whisper.transcribe_audio(str(wav))
    assert "temporarily unavailable" in result.lower()
