import gradio as gr
import numpy as np
import os
import time
import wave
import tempfile
import shutil
import logging
from datetime import datetime
from openai import OpenAI
from pathlib import Path
import subprocess
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('voice_transcriber')

# Get OpenAI API key from environment variable (for Spaces secrets)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class SpacesConfig:
    """Configuration tailored for Hugging Face Spaces"""

    def __init__(self):
        # Use temp directory for recordings to avoid filling Spaces storage
        self.temp_dir = Path(tempfile.gettempdir()) / "spaces_audio"
        self.max_recording_length_seconds = 300  # 5 minutes to avoid timeouts
        self.max_history_items = 5  # Limit history to save memory
        self.supported_languages = [
            "auto", "en", "es", "fr", "de", "it", "pt", "nl",
            "ru", "zh", "ja", "ar", "hi", "ko"
        ]
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
        self.initialize()

    def initialize(self):
        """Initialize directories"""
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self._cleanup_on_start()

    def _cleanup_on_start(self):
        """Clean up old temporary files on startup"""
        try:
            # Hugging Face Spaces starts fresh each time,
            # but we clean up just in case
            for file_path in self.temp_dir.glob("*.*"):
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error during startup cleanup: {e}")

    def _cleanup_on_exit(self):
        """Clean up temporary files on exit"""
        try:
            for file_path in self.temp_dir.glob("*.*"):
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up file on exit: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file on exit {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error during exit cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on exit"""
        try:
            self._cleanup_on_exit()
        except Exception as e:
            logger.error(f"Error in destructor: {e}")


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
        supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "nl",
            "ru", "zh", "ja", "ar", "hi", "ko"
        ]
        return language in supported_languages

    def get_estimated_cost(self, duration_seconds):
        """
        Calculate estimated cost based on current pricing

        Args:
            duration_seconds (float): Duration of audio in seconds

        Returns:
            str: Formatted cost estimate
        """
        # Current pricing: $0.006 per minute
        cost_per_minute = 0.006
        cost = (duration_seconds / 60) * cost_per_minute
        return f"${cost:.4f}"


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


class SpacesTranscriber:
    """Main application class for Hugging Face Spaces"""

    def __init__(self, default_api_key=""):
        self.config = SpacesConfig()
        self.openai_client = OpenAIClient(default_api_key)
        self.audio_processor = AudioProcessor(self.config)
        self.session_history = []
        self.transcription_cache = {}  # Cache for transcriptions

    def connect_to_openai(self, api_key):
        """
        Connect to OpenAI API

        Args:
            api_key (str): The OpenAI API key

        Returns:
            str: HTML-formatted status message
        """
        message, success, color = self.openai_client.connect(api_key)
        return f"<span style='color: {color}'>{message}</span>"

    def handle_recording(self, audio_data, sample_rate, auto_transcribe, api_key, language):
        """
        Process recording and optionally transcribe

        Args:
            audio_data (numpy.ndarray): The audio data
            sample_rate (int): The sample rate in Hz
            auto_transcribe (bool): Whether to automatically transcribe
            api_key (str): OpenAI API key
            language (str): Language code or "auto"

        Returns:
            tuple: A tuple containing:
                - str: Status message
                - str or None: Path to audio file or None
                - str or None: Transcription or None
                - str: Estimated cost
                - float: Duration in seconds
        """
        logger.info(
            f"Processing recording: sample_rate={sample_rate}, auto_transcribe={auto_transcribe}, language={language}")

        # First, process and save the audio
        audio_path, status_msg, duration = self.audio_processor.process_recording(audio_data, sample_rate)

        if not audio_path:
            logger.warning(f"Recording processing failed: {status_msg}")
            return status_msg, None, None, "$0.00", 0

        # Calculate estimated cost
        estimated_cost = self.openai_client.get_estimated_cost(duration)

        # Prepare response
        transcription = None

        # Automatically transcribe if enabled
        if auto_transcribe:
            # Use provided key or default key
            key_to_use = api_key if api_key else self.openai_client.default_api_key

            if key_to_use:
                # Ensure we're connected
                if not self.openai_client.connected:
                    message, success, _ = self.openai_client.connect(key_to_use)
                    if not success:
                        logger.warning(f"Failed to connect to OpenAI: {message}")
                        status_msg += f" (Connection failed: {message})"

                # Transcribe if connected
                if self.openai_client.connected:
                    # Check cache first
                    cache_key = f"{audio_path}_{language}"
                    if cache_key in self.transcription_cache:
                        logger.info(f"Using cached transcription for {audio_path}")
                        transcription = self.transcription_cache[cache_key]
                        status_msg += " (used cached transcription)"
                    else:
                        logger.info(f"Transcribing audio: {audio_path}")
                        transcription = self.openai_client.transcribe_audio(audio_path, language)

                        # Cache successful transcriptions
                        if transcription and not transcription.startswith("Error:") and not transcription.startswith(
                                "Transcription error:"):
                            self.transcription_cache[cache_key] = transcription
                            status_msg += " and transcribed"

                            # Add to session history
                            self.add_to_history(audio_path, transcription, duration)
                        else:
                            logger.warning(f"Transcription failed: {transcription}")
                            status_msg += f" (transcription failed: {transcription})"
            else:
                logger.info("No API key provided for transcription")
                status_msg += " (No API key provided for transcription)"

        logger.info(
            f"Recording handled: path={audio_path}, duration={duration:.2f}s, transcribed={transcription is not None}")
        # Return results
        return status_msg, audio_path, transcription, estimated_cost, duration

    def transcribe_audio(self, audio_path, language, api_key=None):
        """
        Transcribe audio file

        Args:
            audio_path (str): Path to audio file
            language (str): Language code or "auto"
            api_key (str, optional): OpenAI API key

        Returns:
            str: Transcription or error message
        """
        if not audio_path:
            logger.warning("No audio path provided for transcription")
            return "No audio file available for transcription."

        if not os.path.exists(audio_path):
            logger.warning(f"Audio file does not exist: {audio_path}")
            return f"Audio file does not exist: {audio_path}"

        # Check cache first
        cache_key = f"{audio_path}_{language}"
        if cache_key in self.transcription_cache:
            logger.info(f"Using cached transcription for {audio_path}")
            return self.transcription_cache[cache_key]

        # Check API connection
        if not self.openai_client.connected:
            key_to_use = api_key if api_key else self.openai_client.default_api_key
            if key_to_use:
                message, success, _ = self.openai_client.connect(key_to_use)
                if not success:
                    logger.warning(f"Failed to connect to OpenAI: {message}")
                    return f"Not connected to OpenAI API: {message}"
            else:
                logger.warning("No API key provided")
                return "No API key provided. Please enter an OpenAI API key."

        # Now transcribe
        logger.info(f"Transcribing audio: {audio_path}, language: {language}")
        transcription = self.openai_client.transcribe_audio(audio_path, language)

        if transcription and not transcription.startswith("Error:") and not transcription.startswith(
                "Transcription error:"):
            # Cache successful transcription
            self.transcription_cache[cache_key] = transcription

            # Add to session history
            duration = self.audio_processor.recording_length
            self.add_to_history(audio_path, transcription, duration)
            logger.info(f"Transcription successful: {len(transcription)} characters")
        else:
            logger.warning(f"Transcription failed: {transcription}")

        return transcription

    def add_to_history(self, audio_path, transcription, duration):
        """
        Add a recording to the session history

        Args:
            audio_path (str): Path to audio file
            transcription (str): Transcription text
            duration (float): Duration in seconds
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a preview of the transcription
        preview = transcription
        if len(preview) > 100:
            preview = preview[:97] + "..."

        history_item = {
            "timestamp": timestamp,
            "audio_path": audio_path,
            "duration": duration,
            "transcription": preview
        }

        logger.info(f"Adding to history: {timestamp}, duration: {duration:.2f}s, preview: {preview[:30]}...")
        self.session_history.append(history_item)

        # Limit history size for Spaces
        while len(self.session_history) > self.config.max_history_items:
            logger.info(f"History limit reached, removing oldest item")
            self.session_history.pop(0)

    def get_session_history(self):
        """
        Get formatted session history

        Returns:
            str: Formatted history text
        """
        if not self.session_history:
            return "No previous recordings in this session."

        history_text = "### Session History:\n\n"
        for i, item in enumerate(reversed(self.session_history), 1):
            duration_str = time.strftime("%M:%S", time.gmtime(item["duration"]))
            history_text += f"{i}. **{item['timestamp']}** ({duration_str}) - {item['transcription']}\n\n"

        return history_text

    def process_multiple_files(self, file_list, language, api_key):
        """
        Process multiple audio files

        Args:
            file_list (list): List of file paths
            language (str): Language code or "auto"
            api_key (str): OpenAI API key

        Returns:
            list: List of results for each file
        """
        results = []
        for file_path in file_list:
            # Process each file
            processed_path, status, duration = self.audio_processor.process_uploaded_file(file_path)
            if processed_path:
                # Transcribe if processing successful
                transcription = self.transcribe_audio(processed_path, language, api_key)
                results.append({
                    "file": os.path.basename(file_path),
                    "status": status,
                    "transcription": transcription,
                    "duration": duration
                })
            else:
                results.append({
                    "file": os.path.basename(file_path),
                    "status": status,
                    "transcription": "Processing failed",
                    "duration": 0
                })

        return results


# Create the transcriber instance with the default API key
transcriber = SpacesTranscriber(DEFAULT_API_KEY)


def create_interface():
    """Create the Gradio interface for Hugging Face Spaces"""
    with gr.Blocks(title="Voice Recorder & OpenAI Transcriber", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Voice Recorder & OpenAI Transcriber")
        gr.Markdown("Record your voice and transcribe it using OpenAI's Whisper API")

        # Connection status indicator
        if DEFAULT_API_KEY:
            connection_status = gr.Markdown(f"<span style='color: green'>✓ Using environment API key</span>")
        else:
            connection_status = gr.Markdown("⚠️ No API key configured. Please provide one below.")

        # Session state for UI
        current_duration = gr.State(0)

        with gr.Row(equal_height=True, responsive=True) as main_row:
            with gr.Column(scale=6, min_width=500) as main_column:
                # Main content area
                with gr.Group():
                    # OpenAI API Key input and connection
                    with gr.Row():
                        api_key_input = gr.Textbox(
                            label="OpenAI API Key (optional if configured in Spaces)",
                            placeholder="Enter your OpenAI API key here",
                            type="password",
                            container=True,
                            scale=4,
                            value=""
                        )
                        connect_btn = gr.Button("Connect to OpenAI", variant="primary", scale=1)

                    api_status = gr.Markdown("Status: Not connected with custom key")

                with gr.Tabs():
                    with gr.TabItem("Record Audio"):
                        # Audio recording component
                        with gr.Row():
                            audio_recorder = gr.Audio(
                                sources=["microphone"],
                                type="numpy",
                                label="Record Audio",
                                elem_id="audio_recorder",
                                scale=3
                            )

                            with gr.Column(scale=1):
                                auto_transcribe = gr.Checkbox(
                                    label="Auto-transcribe",
                                    value=True,
                                    info="Automatically transcribe after recording"
                                )

                                language_selector = gr.Dropdown(
                                    choices=transcriber.config.supported_languages,
                                    value="auto",
                                    label="Language",
                                    info="Select language or 'auto' for automatic detection"
                                )

                        # Status message and recording info
                        with gr.Row():
                            status_msg = gr.Textbox(
                                label="Status",
                                value="Ready to record",
                                interactive=False,
                                scale=3
                            )
                            cost_display = gr.Textbox(
                                label="Estimated Cost",
                                value="$0.00",
                                interactive=False,
                                scale=1
                            )

                        # Recorded audio playback
                        audio_playback = gr.Audio(
                            label="Recorded Audio",
                            type="filepath",
                            interactive=False,
                            elem_id="audio_playback"
                        )

                        # Manual transcribe button
                        transcribe_btn = gr.Button("Transcribe Audio", variant="secondary")

                    with gr.TabItem("Upload Audio"):
                        # Upload audio file option
                        audio_upload = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"],
                            elem_id="audio_upload"
                        )

                        upload_language = gr.Dropdown(
                            choices=transcriber.config.supported_languages,
                            value="auto",
                            label="Language",
                            info="Select language or 'auto' for automatic detection"
                        )

                        upload_transcribe_btn = gr.Button("Transcribe Uploaded Audio", variant="secondary")
                        upload_status = gr.Textbox(label="Upload Status", value="", interactive=False)

                # Transcription output
                gr.Markdown("### Transcription")
                transcription_output = gr.Textbox(
                    label="Transcription Result",
                    placeholder="Transcription will appear here...",
                    lines=10,
                    max_lines=30,
                    interactive=True,
                    elem_id="transcription_output"
                )

                # Copy and download buttons
                with gr.Row():
                    copy_btn = gr.Button("Copy to Clipboard", elem_id="copy_btn")
                    download_btn = gr.Button("Download Transcript", elem_id="download_btn")

            with gr.Column(scale=3, min_width=300) as side_column:
                # Sidebar-like content
                with gr.Group():
                    gr.Markdown("### Recording Information")

                    # Display recording stats
                    recording_info = gr.Markdown("No recording yet")

                    # Session history
                    gr.Markdown("### Previous Recordings")
                    history_display = gr.Markdown("No previous recordings in this session.")

                with gr.Accordion("Usage Information", open=False):
                    gr.Markdown("""
                    ### How to use this app:

                    1. Record your voice using the microphone or upload an audio file
                    2. The recording will be automatically transcribed if enabled
                    3. Or click "Transcribe Audio" to manually transcribe
                    4. Copy or download the transcription as needed

                    ### API Key:

                    - If this Space has been configured with an API key, you don't need to provide one
                    - Otherwise, enter your OpenAI API key in the field at the top

                    ### Pricing Information:

                    OpenAI's Whisper API costs approximately $0.006 per minute of audio.
                    The cost estimate is displayed after recording.

                    ### Limitations:

                    - Maximum recording length: 5 minutes
                    - Recordings are temporary and not permanently stored
                    - History is limited to the current session
                    """)

                # About this app
                with gr.Accordion("About", open=False):
                    gr.Markdown("""
                    ### About this App

                    This application uses:
                    - Gradio for the web interface
                    - OpenAI's Whisper API for transcription

                    Created for demonstration and educational purposes.

                    _The app is running on Hugging Face Spaces, which provides temporary storage. 
                    Your recordings and transcriptions are not permanently stored._
                    """)

        # Event handlers
        def update_recording_info(duration):
            """Update the recording information display"""
            if duration > 0:
                path, size_kb = transcriber.audio_processor.get_last_recording_info()
                if path:
                    duration_str = time.strftime("%M:%S", time.gmtime(duration))
                    filename = os.path.basename(path)
                    return f"**Last Recording:**\n- Duration: {duration_str}\n- Size: {size_kb:.1f} KB\n- File: {filename}"
            return "No recording yet"

        # Connect to OpenAI
        connect_btn.click(
            fn=transcriber.connect_to_openai,
            inputs=[api_key_input],
            outputs=[api_status]
        )

        # Process recording - IMPROVED: Robust handling of audio data and parameters
        def handle_recording_wrapper(audio_data):
            # Validate audio data format
            if audio_data is None or not isinstance(audio_data, tuple) or len(audio_data) != 2:
                logger.warning(f"Invalid audio data format: {type(audio_data)}")
                return "Invalid or no audio recorded", None, "", "$0.00", 0

            # Extract audio data and sample rate
            audio_array, sample_rate = audio_data

            # Get parameters from UI components safely
            try:
                auto_transcribe_value = auto_transcribe.value
            except:
                logger.warning("Failed to get auto_transcribe value, defaulting to True")
                auto_transcribe_value = True

            try:
                api_key_value = api_key_input.value
            except:
                logger.warning("Failed to get API key value, using default")
                api_key_value = ""

            try:
                language_value = language_selector.value
            except:
                logger.warning("Failed to get language value, defaulting to 'auto'")
                language_value = "auto"

            # Create progress tracker for long transcriptions
            with gr.Progress(track_tqdm=True) as progress_tracker:
                # Call transcriber with all required parameters
                result = transcriber.handle_recording(
                    audio_array,
                    sample_rate,
                    auto_transcribe_value,
                    api_key_value,
                    language_value
                )

            status_msg, audio_path, transcription, estimated_cost, duration = result
            return status_msg, audio_path, transcription or "", estimated_cost, duration

        audio_recorder.stop_recording(
            fn=handle_recording_wrapper,
            inputs=[audio_recorder],  # This provides the audio data
            outputs=[status_msg, audio_playback, transcription_output, cost_display, current_duration]
        ).then(
            fn=update_recording_info,
            inputs=[current_duration],
            outputs=[recording_info]
        ).then(
            fn=transcriber.get_session_history,
            inputs=[],
            outputs=[history_display]
        )

        # Manual transcribe for recorded audio
        def manual_transcribe(path, lang, api_key):
            if not path:
                return "No recording available to transcribe"

            with gr.Progress(track_tqdm=True) as progress_tracker:
                transcription = transcriber.transcribe_audio(path, lang,
                                                             api_key) if path else "No recording available to transcribe"

            return transcription

        transcribe_btn.click(
            fn=manual_transcribe,
            inputs=[audio_playback, language_selector, api_key_input],
            outputs=[transcription_output]
        ).then(
            fn=transcriber.get_session_history,
            inputs=[],
            outputs=[history_display]
        )

        # Transcribe uploaded audio - IMPROVED: Better error handling for different file formats
        def process_uploaded_audio(audio_path, language, api_key):
            if not audio_path:
                return "No file uploaded", ""

            # First process the uploaded file
            with gr.Progress(track_tqdm=True) as progress_tracker:
                progress_tracker.tqdm(0, desc="Processing audio file")
                processed_path, status_msg, duration = transcriber.audio_processor.process_uploaded_file(audio_path)

                if not processed_path:
                    return status_msg, ""

                # Set for cost calculation
                estimated_cost = transcriber.openai_client.get_estimated_cost(duration)

                # Update progress
                progress_tracker.tqdm(0.5, desc="Transcribing audio")

                # Transcribe
                transcription = transcriber.transcribe_audio(processed_path, language, api_key)

            status_with_cost = f"{status_msg} (Estimated cost: {estimated_cost})"
            return status_with_cost, transcription or ""

        upload_transcribe_btn.click(
            fn=process_uploaded_audio,
            inputs=[audio_upload, upload_language, api_key_input],
            outputs=[upload_status, transcription_output]
        ).then(
            fn=transcriber.get_session_history,
            inputs=[],
            outputs=[history_display]
        )

        # IMPROVED: Copy to clipboard with proper JS integration
        def copy_to_clipboard(text):
            return None

        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[transcription_output],
            outputs=[],
            js="""
            async (text) => {
                if (!text) {
                    // Show message if there's nothing to copy
                    const el = document.getElementById('copy_btn');
                    const originalText = el.textContent;
                    el.textContent = 'Nothing to copy!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                    return [];
                }

                try {
                    await navigator.clipboard.writeText(text);
                    // Show a brief message
                    const el = document.getElementById('copy_btn');
                    const originalText = el.textContent;
                    el.textContent = '✓ Copied!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                } catch (err) {
                    console.error('Failed to copy: ', err);
                    // Show error message
                    const el = document.getElementById('copy_btn');
                    const originalText = el.textContent;
                    el.textContent = '❌ Copy failed!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                }
                return [];
            }
            """
        )

        # IMPROVED: Download transcript with proper JS integration
        def download_transcript(text):
            return None

        download_btn.click(
            fn=download_transcript,
            inputs=[transcription_output],
            outputs=[],
            js="""
            async (text) => {
                if (!text) {
                    // Show message if there's nothing to download
                    const el = document.getElementById('download_btn');
                    const originalText = el.textContent;
                    el.textContent = 'Nothing to download!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                    return [];
                }

                try {
                    const blob = new Blob([text], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `transcript_${new Date().toISOString().slice(0,19).replace(/[:.]/g, '-')}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);

                    // Show a brief message
                    const el = document.getElementById('download_btn');
                    const originalText = el.textContent;
                    el.textContent = '✓ Downloaded!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                } catch (err) {
                    console.error('Failed to download: ', err);
                    // Show error message
                    const el = document.getElementById('download_btn');
                    const originalText = el.textContent;
                    el.textContent = '❌ Download failed!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                }
                return [];
            }
            """
        )

    return app


# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch()