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
            for file_path in self.temp_dir.glob("*.wav"):
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up file: {file_path}")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error during startup cleanup: {e}")


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
        """Initialize the OpenAI client with the provided API key"""
        if not api_key:
            return "Please enter your OpenAI API key", False, "red"

        try:
            client = OpenAI(api_key=api_key)
            # Test connection with a lightweight call
            client.models.list()
            self.client = client
            self.connected = True
            self.last_error = None
            return "✓ Successfully connected to OpenAI API", True, "green"
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI connection error: {error_msg}")
            return f"❌ Error connecting to OpenAI: {error_msg}", False, "red"

    def transcribe_audio(self, audio_path, language="auto"):
        """Transcribe audio using OpenAI's Whisper API"""
        if not self.client:
            return "OpenAI client not initialized. Please enter your API key."

        if not audio_path or not os.path.exists(audio_path):
            return "No audio file available for transcription."

        try:
            # Additional parameters based on language
            params = {"model": "whisper-1"}
            if language and language != "auto":
                params["language"] = language

            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    **params
                )

            return response.text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Transcription error: {error_msg}")
            return f"Transcription error: {error_msg}"

    def get_estimated_cost(self, duration_seconds):
        """Calculate estimated cost based on current pricing"""
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
        """Process and save the recorded audio data"""
        if audio_data is None:
            return None, "No audio recorded", 0

        # Check recording length
        duration_seconds = len(audio_data) / sample_rate
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
            return temp_path, f"Recording saved ({self._format_duration(duration_seconds)})", duration_seconds

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error saving audio: {error_msg}")
            return None, f"Error saving audio: {error_msg}", 0

    def _format_duration(self, seconds):
        """Format duration in seconds to mm:ss format"""
        return time.strftime("%M:%S", time.gmtime(seconds))

    def get_last_recording_info(self):
        """Get information about the last recording"""
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

    def connect_to_openai(self, api_key):
        """Connect to OpenAI API"""
        message, success, color = self.openai_client.connect(api_key)
        return f"<span style='color: {color}'>{message}</span>"

    def handle_recording(self, audio_data, sample_rate, auto_transcribe, api_key, language):
        """Process recording and optionally transcribe"""
        # First, process and save the audio
        audio_path, status_msg, duration = self.audio_processor.process_recording(audio_data, sample_rate)

        if not audio_path:
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

                # Transcribe if connected
                if self.openai_client.connected:
                    transcription = self.openai_client.transcribe_audio(audio_path, language)
                    if transcription and not transcription.startswith("Error:") and not transcription.startswith(
                            "Transcription error:"):
                        status_msg += " and transcribed"

                        # Add to session history
                        self.add_to_history(audio_path, transcription, duration)
            else:
                status_msg += " (No API key provided for transcription)"

        # Return results
        return status_msg, audio_path, transcription, estimated_cost, duration

    def transcribe_audio(self, audio_path, language, api_key=None):
        """Transcribe audio file"""
        if not audio_path or not os.path.exists(audio_path):
            return "No audio file available for transcription."

        # Check API connection
        if not self.openai_client.connected:
            key_to_use = api_key if api_key else self.openai_client.default_api_key
            if key_to_use:
                message, success, _ = self.openai_client.connect(key_to_use)
                if not success:
                    return f"Not connected to OpenAI API: {message}"
            else:
                return "No API key provided. Please enter an OpenAI API key."

        # Now transcribe
        transcription = self.openai_client.transcribe_audio(audio_path, language)

        if transcription and not transcription.startswith("Error:") and not transcription.startswith(
                "Transcription error:"):
            # Add to session history
            duration = self.audio_processor.recording_length
            self.add_to_history(audio_path, transcription, duration)

        return transcription

    def add_to_history(self, audio_path, transcription, duration):
        """Add a recording to the session history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a preview of the transcription
        preview = transcription
        if len(preview) > 100:
            preview = preview[:97] + "..."

        self.session_history.append({
            "timestamp": timestamp,
            "audio_path": audio_path,
            "duration": duration,
            "transcription": preview
        })

        # Limit history size for Spaces
        while len(self.session_history) > self.config.max_history_items:
            self.session_history.pop(0)

    def get_session_history(self):
        """Get formatted session history"""
        if not self.session_history:
            return "No previous recordings in this session."

        history_text = "### Session History:\n\n"
        for i, item in enumerate(reversed(self.session_history), 1):
            duration_str = time.strftime("%M:%S", time.gmtime(item["duration"]))
            history_text += f"{i}. **{item['timestamp']}** ({duration_str}) - {item['transcription']}\n\n"

        return history_text


# Create the transcriber instance with the default API key
transcriber = SpacesTranscriber(DEFAULT_API_KEY)


def create_interface():
    """Create the Gradio interface for Hugging Face Spaces"""
    with gr.Blocks(title="Voice Recorder & OpenAI Transcriber") as app:
        gr.Markdown("# 🎙️ Voice Recorder & OpenAI Transcriber")
        gr.Markdown("Record your voice and transcribe it using OpenAI's Whisper API")

        # Connection status indicator
        if DEFAULT_API_KEY:
            connection_status = gr.Markdown(f"<span style='color: green'>✓ Using environment API key</span>")
        else:
            connection_status = gr.Markdown("⚠️ No API key configured. Please provide one below.")

        # Session state for UI
        current_duration = gr.State(0)

        with gr.Row():
            with gr.Column(scale=6):
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

            with gr.Column(scale=3):
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

        # Process recording
        def handle_recording_wrapper(audio_data, sample_rate, auto_transcribe, api_key, language):
            result = transcriber.handle_recording(audio_data, sample_rate, auto_transcribe, api_key, language)
            status_msg, audio_path, transcription, estimated_cost, duration = result
            return status_msg, audio_path, transcription or "", estimated_cost, duration

        audio_recorder.stop_recording(
            fn=handle_recording_wrapper,
            inputs=[audio_recorder, auto_transcribe, api_key_input, language_selector],
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
        transcribe_btn.click(
            fn=lambda path, lang, api_key: transcriber.transcribe_audio(path, lang,
                                                                        api_key) if path else "No recording available to transcribe",
            inputs=[audio_playback, language_selector, api_key_input],
            outputs=[transcription_output]
        ).then(
            fn=transcriber.get_session_history,
            inputs=[],
            outputs=[history_display]
        )

        # Transcribe uploaded audio
        def process_uploaded_audio(audio_path, language, api_key):
            if not audio_path:
                return "No file uploaded", ""

            # Set for cost calculation
            try:
                # Get audio duration
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)

                # Set for cost calculation
                transcriber.audio_processor.recording_length = duration
                transcriber.audio_processor.last_recording_path = audio_path

                # Transcribe
                transcription = transcriber.transcribe_audio(audio_path, language, api_key)
                return f"Uploaded file transcribed ({time.strftime('%M:%S', time.gmtime(duration))})", transcription or ""
            except Exception as e:
                return f"Error processing uploaded file: {str(e)}", ""

        upload_transcribe_btn.click(
            fn=process_uploaded_audio,
            inputs=[audio_upload, upload_language, api_key_input],
            outputs=[upload_status, transcription_output]
        ).then(
            fn=transcriber.get_session_history,
            inputs=[],
            outputs=[history_display]
        )

        # Copy to clipboard
        copy_btn.click(
            fn=None,
            inputs=[transcription_output],
            outputs=[],
            _js="""
            function(text) {
                navigator.clipboard.writeText(text);
                // Show a brief message
                const el = document.getElementById('copy_btn');
                const originalText = el.textContent;
                el.textContent = '✓ Copied!';
                setTimeout(() => { el.textContent = originalText; }, 2000);
                return [];
            }
            """
        )

        # Download transcript
        download_btn.click(
            fn=None,
            inputs=[transcription_output],
            outputs=[],
            _js="""
            function(text) {
                if (!text) return [];

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
                return [];
            }
            """
        )

    return app


# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch()