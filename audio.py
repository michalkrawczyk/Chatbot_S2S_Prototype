import os
import subprocess
import shutil
import tempfile
import time

from general.logs import logger

import numpy as np
import wave

class AudioProcessor:
    """Handles audio recording and processing"""

    def __init__(self, config):
        self.config = config
        self.last_recording_path = None
        self.recording_length = 0

    def process_recording(self, audio_data, sample_rate):
        """
        Process and save the recorded audio data

        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio in Hz

        Returns:
            tuple: A tuple containing:
                - str or None: Path to saved audio file or None if processing failed
                - str: Status message describing the result
                - float: Duration of the recording in seconds

        Raises:
            ValueError: If audio_data is invalid or corrupted
        """
        if audio_data is None:
            logger.warning("No audio data received")
            return None, "No audio recorded", 0

        try:
            # Check recording length
            duration_seconds = len(audio_data) / sample_rate
            logger.info(f"Processing recording: {duration_seconds:.2f}s at {sample_rate}Hz")

            if duration_seconds > self.config.max_recording_length_seconds:
                logger.warning(f"Recording too long: {duration_seconds}s > {self.config.max_recording_length_seconds}s")
                return None, f"Recording too long (max: {self.config.max_recording_length_seconds}s)", 0

            # Create a temp file with a unique name
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=self.config.temp_dir, delete=False) as tmp:
                temp_path = tmp.name

            # Calculate recording length
            self.recording_length = duration_seconds

            # Save the audio file
            try:
                # Convert to int16 format for better compatibility with Whisper
                audio_data_int16 = (audio_data * 32767).astype(np.int16)

                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 2 bytes for int16
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data_int16.tobytes())

                self.last_recording_path = temp_path
                logger.info(f"Recording saved to {temp_path}")
                return temp_path, f"Recording saved ({self._format_duration(duration_seconds)})", duration_seconds

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error saving audio: {error_msg}")
                # Clean up the temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                return None, f"Error saving audio: {error_msg}", 0

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing audio: {error_msg}")
            return None, f"Error processing audio: {error_msg}", 0

    def process_uploaded_file(self, file_path):
        """
        Process an uploaded audio file

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            tuple: A tuple containing:
                - str or None: Path to processed audio file or None if processing failed
                - str: Status message describing the result
                - float: Duration of the audio in seconds
        """
        if not file_path or not os.path.exists(file_path):
            logger.warning("No file uploaded or file doesn't exist")
            return None, "No file uploaded or file doesn't exist", 0

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.config.supported_formats:
            supported_formats_str = ", ".join(self.config.supported_formats)
            logger.warning(f"Unsupported file format: {file_ext}. Supported formats: {supported_formats_str}")
            return None, f"Unsupported file format. Please upload {supported_formats_str}", 0

        try:
            # If not WAV, convert to WAV for better compatibility with Whisper
            if file_ext != '.wav':
                logger.info(f"Converting {file_ext} to WAV format")
                converted_path = self._convert_to_wav(file_path)
                if not converted_path:
                    return None, f"Error converting {file_ext} to WAV format", 0
                processed_path = converted_path
            else:
                # Create a copy in our temp directory
                processed_path = os.path.join(self.config.temp_dir, os.path.basename(file_path))
                shutil.copy2(file_path, processed_path)

            # Get audio duration
            with wave.open(processed_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)

            # Check if duration exceeds maximum
            if duration > self.config.max_recording_length_seconds:
                logger.warning(f"Uploaded file too long: {duration}s > {self.config.max_recording_length_seconds}s")
                return None, f"Uploaded file too long (max: {self.config.max_recording_length_seconds}s)", 0

            # Store file info
            self.last_recording_path = processed_path
            self.recording_length = duration

            logger.info(f"Uploaded file processed: {processed_path}, duration: {duration:.2f}s")
            return processed_path, f"File processed ({self._format_duration(duration)})", duration

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing uploaded file: {error_msg}")
            return None, f"Error processing uploaded file: {error_msg}", 0

    def _convert_to_wav(self, input_path):
        """
        Convert audio file to WAV format using ffmpeg

        Args:
            input_path (str): Path to input audio file

        Returns:
            str or None: Path to converted WAV file or None if conversion failed
        """
        try:
            # Generate output path
            output_path = os.path.join(
                self.config.temp_dir,
                f"{os.path.splitext(os.path.basename(input_path))[0]}.wav"
            )

            # Use ffmpeg if available, otherwise use subprocess
            try:
                import ffmpy
                ff = ffmpy.FFmpeg(
                    inputs={input_path: None},
                    outputs={output_path: '-ac 1 -ar 16000'}
                )
                ff.run()
            except ImportError:
                # Fallback to subprocess
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-ac', '1', '-ar', '16000',
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            if os.path.exists(output_path):
                return output_path
            return None

        except Exception as e:
            logger.error(f"Error converting file to WAV: {str(e)}")
            return None

    def _format_duration(self, seconds):
        """
        Format duration in seconds to mm:ss format

        Args:
            seconds (float): Duration in seconds

        Returns:
            str: Formatted duration string
        """
        return time.strftime("%M:%S", time.gmtime(seconds))

    def get_last_recording_info(self):
        """
        Get information about the last recording

        Returns:
            tuple: A tuple containing:
                - str or None: Path to the last recording or None if no recording
                - float: File size in KB
        """
        if not self.last_recording_path or not os.path.exists(self.last_recording_path):
            return None, 0

        file_size_kb = os.path.getsize(self.last_recording_path) / 1024  # KB
        return self.last_recording_path, file_size_kb