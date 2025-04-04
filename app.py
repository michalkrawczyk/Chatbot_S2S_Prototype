import gradio as gr

import os
import time
import tempfile
import shutil
import atexit

from datetime import datetime

from pathlib import Path

from agents import AgentLLM
from utils import logger, conditional_debug_info
from audio import AudioProcessor
from openai_client import OpenAIClient, SUPPORT_LANGUAGES

import traceback




# Get OpenAI API key from environment variable (for Spaces secrets)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

AGENT = AgentLLM()


class SpacesConfig:
    """Configuration tailored for Hugging Face Spaces"""

    def __init__(self):
        # Use temp directory for recordings to avoid filling Spaces storage
        self.temp_dir = Path(tempfile.gettempdir()) / "spaces_audio"
        self.memory_files_dir = Path("memory_files")  # New directory for uploaded files
        self.max_recording_length_seconds = 300  # 5 minutes to avoid timeouts
        self.max_history_items = 5  # Limit history to save memory
        self.supported_languages = SUPPORT_LANGUAGES
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
        self.tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]  # TTS voice options
        self.initialize()

    def initialize(self):
        """Initialize directories"""
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.memory_files_dir.mkdir(exist_ok=True, parents=True)  # Create memory_files directory
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


class SpacesTranscriber:
    """Main application class for Hugging Face Spaces"""

    def __init__(self, default_api_key=""):
        self.config = SpacesConfig()
        self.openai_client = OpenAIClient(default_api_key)
        self.audio_processor = AudioProcessor(self.config)
        self.session_history = []
        self.transcription_cache = {}  # Cache for transcriptions
        self.agent_memory = []  # Memory for the agent
        self.thinking_process = ""  # For storing agent's thinking process
        self.current_model = None  # Track the current model
        self.default_response_language = "eng" # Default language for responses (For now, English)

    def connect_to_openai(self, api_key):
        """
        Connect to OpenAI API

        Args:
            api_key (str): The OpenAI API key

        Returns:
            str: HTML-formatted status message
        """
        message, success, color = self.openai_client.connect(api_key)
        status_html = f"<span style='color: {color}'>{message}</span>"

        if success:
            # Initialize the agent with the default model
            agent_init_success = self.initialize_agent_ui(api_key, model_name=self.current_model,
                                                          target_language=self.default_response_language)

            # Pass agent status to UI
            if agent_init_success:
                return status_html, f"Agent Status: ✓ Initialized with {self.current_model} model"
            else:
                return status_html, "Agent Status: ❌ Initialization failed"

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
                - float: Duration in seconds
        """
        logger.info(
            f"Processing recording: sample_rate={sample_rate}, auto_transcribe={auto_transcribe}, language={language}")

        # First, process and save the audio
        audio_path, status_msg, duration = self.audio_processor.process_recording(audio_data, sample_rate)

        if not audio_path:
            logger.warning(f"Recording processing failed: {status_msg}")
            return status_msg, None, None, 0

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
        return status_msg, audio_path, transcription, duration

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

    def initialize_agent_ui(self, api_key, model_name=None, target_language="eng"):
        """Initialize the agent with the OpenAI API key and optional model."""
        if model_name:
            self.current_model = model_name
            logger.info(f"Setting agent model to: {model_name}")

        return AGENT.initialize_agent(api_key, model_name=self.current_model, target_language=target_language)

    def analyze_transcription(self, transcription, source_file=None):
        """
        Analyze the transcription using the AI agent.

        Args:
            transcription (str): The text to analyze
            source_file (str, optional): Source file for additional context

        Returns:
            tuple: (analysis_result, thinking_process)
        """
        if not AGENT.get_agent_executor:
            return "Agent not initialized. Please provide a valid API key.", ""

        context = transcription
        if source_file:
            try:
                with open(source_file, 'r') as f:
                    file_content = f.read()
                context = f"Transcription: {transcription}\n\nSource File Content: {file_content}"
            except Exception as e:
                logger.error(f"Error reading source file: {e}")
                context = f"Transcription: {transcription}\n\nSource File: [Failed to read file: {str(e)}]"

        result, thinking = AGENT.run_agent_on_text(context, self.agent_memory, return_thinking=True)
        self.thinking_process = thinking
        return result, thinking

    def clear_agent_memory(self):
        """Clear the agent's memory."""
        self.agent_memory = []
        return "Agent memory cleared."

    def add_to_agent_memory(self, transcription, analysis):
        """Add the current context to agent memory."""
        if not hasattr(self, 'agent_memory'):
            self.agent_memory = []

        memory_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcription": transcription,
            "analysis": analysis
        }

        self.agent_memory.append(memory_item)

        # Limit memory size
        max_memory_items = 10
        if len(self.agent_memory) > max_memory_items:
            self.agent_memory = self.agent_memory[-max_memory_items:]

    def generate_speech_from_text(self, text, voice="alloy"):
        """
        Generate speech from text using OpenAI's TTS API

        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use

        Returns:
            tuple: (audio_data or path or None, status message)
        """
        if not text:
            return None, "No text provided for speech synthesis"

        if not self.openai_client.connected:
            return None, "Not connected to OpenAI API"

        try:
            # Limit text length to avoid timeouts
            max_chars = 4000
            if len(text) > max_chars:
                text = text[:max_chars] + "... (text truncated for speech synthesis)"

            # Get audio from OpenAI - this returns bytes directly since stream=True is the default
            result = self.openai_client.text_to_speech(text, voice)

            if isinstance(result, str) and result.startswith("Error"):
                return None, result

            # For Gradio's audio component to work with bytes, we need to save the data
            # to a temporary file since Gradio's Audio component works best with files
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name

            # Write bytes to file if we got bytes back
            if isinstance(result, bytes):
                with open(temp_path, 'wb') as f:
                    f.write(result)
                # Register for cleanup
                atexit.register(lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
                return temp_path, "Speech generated successfully"

            # If we got a path back (stream=False was used), just return it
            return result, "Speech generated successfully"

        except Exception as e:
            logger.error(f"Error in speech generation: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None, f"Error generating speech: {str(e)}"


# Create the transcriber instance with the default API key
transcriber = SpacesTranscriber(DEFAULT_API_KEY)


def create_interface():
    """Create the Gradio interface for Hugging Face Spaces"""
    with gr.Blocks(title="Voice Recorder & AI Analysis") as app:
        gr.Markdown("# 🎙️ Voice Recorder & AI Analysis")
        gr.Markdown("Record your voice, transcribe it, and analyze it with AI")

        # Connection status indicator
        if DEFAULT_API_KEY:
            connection_status = gr.Markdown(f"<span style='color: green'>✓ Using environment API key</span>")
        else:
            connection_status = gr.Markdown("⚠️ No API key configured. Please provide one below.")

        # Session state for UI
        current_duration = gr.State(0)
        source_file_path = gr.State(None)

        with gr.Row() as main_row:
            # API Key configuration
            with gr.Column(scale=1):
                with gr.Group():
                    # OpenAI API Key input and connection
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="Enter your OpenAI API key here",
                        type="password",
                        container=True,
                        value=""
                    )
                    connect_btn = gr.Button("Connect to OpenAI", variant="primary")
                    api_status = gr.Markdown("Status: Not connected with custom key")

                    # Agent model selector
                    agent_model_selector = gr.Dropdown(
                        choices=["o3-mini", "gpt-4-turbo", "gpt-4o"],
                        value="o3-mini",
                        label="Agent Model",
                        info="Select the model for the AI agent"
                    )
                    agent_status = gr.Markdown("Agent Status: Not initialized")

                # Language selection
                language_selector = gr.Dropdown(
                    choices=transcriber.config.supported_languages,
                    value="auto",
                    label="Language",
                    info="Select language or 'auto' for automatic detection"
                )

                # TTS voice selection
                voice_selector = gr.Dropdown(
                    choices=transcriber.config.tts_voices,
                    value="alloy",
                    label="TTS Voice",
                    info="Select voice for text-to-speech"
                )

                # Auto-speak option
                auto_speak = gr.Checkbox(
                    label="Auto-speak analysis",
                    value=True,
                    info="Automatically generate speech for analysis results"
                )

                # File upload section for memory files
                with gr.Group():
                    gr.Markdown("### Upload Files to Memory")
                    memory_files_upload = gr.File(
                        label="Upload Files",
                        file_count="multiple",
                        type="filepath",
                        file_types=[".txt", ".pdf", ".docx", ".jpg", ".png", ".csv", ".xls", ".xlsx"]
                    )
                    upload_memory_files_btn = gr.Button("Save to Memory Files", variant="secondary")
                    memory_upload_status = gr.Textbox(label="Upload Status", value="", interactive=False)

                # Display uploaded files
                with gr.Accordion("Uploaded Files", open=True):
                    uploaded_files_display = gr.Markdown("No files have been uploaded yet.")
                    refresh_files_btn = gr.Button("Refresh Files List", variant="secondary")

                # Agent memory
                with gr.Accordion("Agent Memory", open=False):
                    memory_display = gr.Markdown("No memory stored.")
                    clear_memory_btn = gr.Button("Clear Memory", variant="secondary")

            # Main content area
            with gr.Column(scale=3):
                with gr.Tabs() as input_tabs:
                    # Tab 1: Record Audio
                    with gr.TabItem("Record Audio"):
                        audio_recorder = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record Audio",
                            elem_id="audio_recorder"
                        )
                        audio_playback = gr.Audio(
                            label="Recorded Audio",
                            type="filepath",
                            interactive=False,
                            elem_id="audio_playback"
                        )
                        auto_transcribe = gr.Checkbox(
                            label="Auto-transcribe & Analyze",
                            value=True,
                            info="Automatically transcribe and analyze after recording"
                        )

                    # Tab 2: Upload Audio
                    with gr.TabItem("Upload Audio"):
                        audio_upload = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"],
                            elem_id="audio_upload"
                        )
                        upload_status = gr.Textbox(label="Upload Status", value="", interactive=False)
                        upload_transcribe_btn = gr.Button("Transcribe & Analyze Uploaded Audio", variant="secondary")

                    # Tab 3: Text Input
                    with gr.TabItem("Text Input"):
                        text_input = gr.Textbox(
                            label="Enter Text for Analysis",
                            placeholder="Type or paste text here for AI analysis...",
                            lines=10,
                            max_lines=30
                        )
                        analyze_text_btn = gr.Button("Analyze Text", variant="primary")

                # Status info
                status_msg = gr.Textbox(
                    label="Status",
                    value="Ready to record or input text",
                    interactive=False
                )

                # Transcription output
                gr.Markdown("### Transcription")
                transcription_output = gr.Textbox(
                    label="Transcribed Text",
                    placeholder="Transcription will appear here...",
                    lines=6,
                    max_lines=10,
                    interactive=True
                )

                # Copy and download buttons for transcription
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Transcription", variant="primary")

            # Output area for analysis
            with gr.Column(scale=2):
                # Analysis output
                gr.Markdown("### AI Analysis")
                analysis_output = gr.Textbox(
                    label="Analysis Result",
                    placeholder="Analysis will appear here...",
                    lines=15,
                    max_lines=25,
                    interactive=True
                )

                # TTS playback and control
                speak_analysis_btn = gr.Button("🔊 Speak Analysis", variant="secondary")
                tts_audio = gr.Audio(
                    label="Analysis Audio",
                    type="filepath",
                    interactive=False
                )

                # Thinking process display
                gr.Markdown("### Agent Thinking Process")
                thinking_display = gr.Textbox(
                    label="Agent's Thinking Process",
                    placeholder="The agent's reasoning will appear here...",
                    lines=15,
                    max_lines=25,
                    interactive=True
                )

                # Copy analysis button
                copy_analysis_btn = gr.Button("Copy Analysis", elem_id="copy_analysis_btn")

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
        def connect_to_openai(api_key):
            message, success, color = transcriber.openai_client.connect(api_key)
            status_html = f"<span style='color: {color}'>{message}</span>"
            return status_html

        connect_btn.click(
            fn=connect_to_openai,
            inputs=[api_key_input],
            outputs=[api_status]
        )

        # Auto-initialize agent when model is selected
        def on_model_change(model, api_key):
            if not api_key:
                return "Agent Status: ⚠️ No API key provided"

            # Ensure we're connected to OpenAI
            if not transcriber.openai_client.connected:
                message, success, _ = transcriber.openai_client.connect(api_key)
                if not success:
                    return f"Agent Status: ❌ Connection failed: {message}"

            # Initialize agent with selected model
            success = transcriber.initialize_agent_ui(api_key, model, target_language=language_selector.value)
            #TODO: ADD agent reinitialization on language change
            if success:
                conditional_debug_info(
                    f"Agent initialized with model: {model}, is correct: {AgentLLM.get_agent_executor is not None}")
                return f"Agent Status: ✓ Initialized with {model} model"
            else:
                return "Agent Status: ❌ Initialization failed"

        agent_model_selector.change(
            fn=on_model_change,
            inputs=[agent_model_selector, api_key_input],
            outputs=[agent_status]
        )

        # Process recording with auto transcription and analysis
        def handle_recording_with_analysis(audio_data):
            # Validate audio data format
            if audio_data is None or not isinstance(audio_data, tuple) or len(audio_data) != 2:
                logger.warning(f"Invalid audio data format: {type(audio_data)}")
                return "Invalid or no audio recorded", None, "", 0, "", "", None

            # Extract audio data and sample rate
            audio_array, sample_rate = audio_data

            # Get parameters from UI components safely
            try:
                auto_analyze_value = auto_transcribe.value
            except:
                logger.warning("Failed to get auto_transcribe value, defaulting to True")
                auto_analyze_value = True

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

            # Get TTS settings
            try:
                auto_speak_value = auto_speak.value
                voice_value = voice_selector.value
            except:
                logger.warning("Failed to get TTS settings, using defaults")
                auto_speak_value = False
                voice_value = "alloy"

            # Call transcriber
            try:
                result = transcriber.handle_recording(
                    audio_array,
                    sample_rate,
                    True,  # Always transcribe
                    api_key_value,
                    language_value
                )

                status_msg, audio_path, transcription, duration = result

                # Default values in case analysis fails
                analysis_result = ""
                thinking_process = ""
                tts_audio_path = None

                # Automatically analyze if enabled
                if auto_analyze_value and transcription and not transcription.startswith("Error"):
                    try:
                        analysis_result, thinking_process = transcriber.analyze_transcription(transcription)
                        # Add to agent memory
                        transcriber.add_to_agent_memory(transcription, analysis_result)
                        status_msg += " and analyzed"

                        # Generate speech if auto-speak is enabled
                        if auto_speak_value:
                            tts_audio_path, tts_status = transcriber.generate_speech_from_text(analysis_result,
                                                                                               voice_value)
                            if tts_audio_path:
                                status_msg += " with speech"
                            else:
                                status_msg += f" (speech generation failed: {tts_status})"
                    except Exception as e:
                        logger.error(f"Error analyzing transcription: {e}")
                        analysis_result = f"Analysis error: {str(e)}"
                        thinking_process = ""

                return status_msg, audio_path, transcription or "", duration, analysis_result, thinking_process, tts_audio_path
            except Exception as e:
                logger.error(f"Error in handle_recording_wrapper: {str(e)}")
                return f"Error processing recording: {str(e)}", None, "", 0, "", "", None

        audio_recorder.stop_recording(
            fn=handle_recording_with_analysis,
            inputs=[audio_recorder],  # Only the audio recorder data is passed
            outputs=[status_msg, audio_playback, transcription_output, current_duration, analysis_output,
                     thinking_display, tts_audio]
        ).then(
            fn=update_recording_info,
            inputs=[current_duration],
            outputs=[status_msg]
        )

        # Process uploaded audio with automatic analysis
        def process_uploaded_audio_with_analysis(audio_path, language, api_key):
            if not audio_path:
                return "No file uploaded", "", "", "", None

            # Get TTS settings
            try:
                auto_speak_value = auto_speak.value
                voice_value = voice_selector.value
            except:
                logger.warning("Failed to get TTS settings, using defaults")
                auto_speak_value = False
                voice_value = "alloy"

            try:
                # Process the uploaded file
                logger.info(f"Processing uploaded audio: {audio_path}")
                processed_path, status_msg, duration = transcriber.audio_processor.process_uploaded_file(audio_path)
                if not processed_path:
                    return status_msg, "", "", "", None
                conditional_debug_info(f"- Result: {processed_path}, {status_msg}, {duration:.2f}s")

                logger.info(f"Transcribing audio file: {processed_path}, duration: {duration:.2f}s")
                # Transcribe
                transcription = transcriber.transcribe_audio(processed_path, language, api_key)
                conditional_debug_info(f"- Transcription: {transcription}")

                # Analyze
                analysis_result = ""
                thinking_process = ""
                tts_audio_path = None

                if transcription and not transcription.startswith("Error"):
                    analysis_result, thinking_process = transcriber.analyze_transcription(transcription)
                    conditional_debug_info(f"- Analysis: {analysis_result}")
                    conditional_debug_info(f"- Thinking: {thinking_process}")
                    # Add to agent memory
                    transcriber.add_to_agent_memory(transcription, analysis_result)

                    # Generate speech if auto-speak is enabled
                    if auto_speak_value:
                        tts_audio_path, tts_status = transcriber.generate_speech_from_text(analysis_result, voice_value)
                        if not tts_audio_path:
                            status_msg += f" (speech generation failed: {tts_status})"

                conditional_debug_info(
                    f"status_msg: {status_msg}, transcription: {transcription}, analysis_result: {analysis_result}")
                return status_msg, transcription or "", analysis_result, thinking_process, tts_audio_path
            except Exception as e:
                logger.error(f"Error processing uploaded audio: {str(e)}")
                conditional_debug_info(traceback.format_exc())
                return f"Error processing file: {str(e)}", "", "", "", None

        upload_transcribe_btn.click(
            fn=process_uploaded_audio_with_analysis,
            inputs=[audio_upload, language_selector, api_key_input],
            outputs=[upload_status, transcription_output, analysis_output, thinking_display, tts_audio]
        )

        # Analyze text input
        def analyze_text_input(text):
            if not text:
                return "No text provided for analysis", "", "", None

            # Get TTS settings
            try:
                auto_speak_value = auto_speak.value
                voice_value = voice_selector.value
            except:
                logger.warning("Failed to get TTS settings, using defaults")
                auto_speak_value = False
                voice_value = "alloy"

            try:
                # Analyze text (no source file)
                analysis_result, thinking_process = transcriber.analyze_transcription(text)

                # Add to agent memory
                transcriber.add_to_agent_memory(text, analysis_result)

                # Generate speech if auto-speak is enabled
                tts_audio_path = None
                if auto_speak_value:
                    tts_audio_path, tts_status = transcriber.generate_speech_from_text(analysis_result, voice_value)
                    if not tts_audio_path:
                        return f"Text analysis completed but speech generation failed: {tts_status}", analysis_result, thinking_process, None

                return "Text analysis completed", analysis_result, thinking_process, tts_audio_path
            except Exception as e:
                logger.error(f"Error analyzing text: {str(e)}")
                conditional_debug_info(traceback.format_exc())
                return f"Error analyzing text: {str(e)}", "", "", None

        analyze_text_btn.click(
            fn=analyze_text_input,
            inputs=[text_input],
            outputs=[status_msg, analysis_output, thinking_display, tts_audio]
        ).then(
            fn=lambda: update_memory_display(),
            inputs=[],
            outputs=[memory_display]
        )

        # Analyze transcription
        def analyze_current_transcription(transcription):
            if not transcription:
                return "No transcription to analyze", "", "", None

            # Get TTS settings
            try:
                auto_speak_value = auto_speak.value
                voice_value = voice_selector.value
            except:
                logger.warning("Failed to get TTS settings, using defaults")
                auto_speak_value = False
                voice_value = "alloy"

            try:
                analysis_result, thinking_process = transcriber.analyze_transcription(transcription)

                # Add to agent memory
                transcriber.add_to_agent_memory(transcription, analysis_result)

                # Generate speech if auto-speak is enabled
                tts_audio_path = None
                if auto_speak_value:
                    tts_audio_path, tts_status = transcriber.generate_speech_from_text(analysis_result, voice_value)
                    if not tts_audio_path:
                        return f"Analysis completed but speech generation failed: {tts_status}", analysis_result, thinking_process, None

                return "Analysis completed", analysis_result, thinking_process, tts_audio_path
            except Exception as e:
                logger.error(f"Error in analyze_transcription: {str(e)}")
                return f"Analysis error: {str(e)}", "", "", None

        analyze_btn.click(
            fn=analyze_current_transcription,
            inputs=[transcription_output],
            outputs=[status_msg, analysis_output, thinking_display, tts_audio]
        ).then(
            fn=lambda: update_memory_display(),
            inputs=[],
            outputs=[memory_display]
        )

        # Speech analysis
        def speak_analysis(analysis_text, voice):
            if not analysis_text:
                return "No analysis to speak", None

            try:
                audio_path, status = transcriber.generate_speech_from_text(analysis_text, voice)
                return status, audio_path
            except Exception as e:
                logger.error(f"Error in speech generation: {str(e)}")
                return f"Speech generation error: {str(e)}", None

        speak_analysis_btn.click(
            fn=speak_analysis,
            inputs=[analysis_output, voice_selector],
            outputs=[status_msg, tts_audio]
        )

        # Handle file uploads to memory_files
        def save_to_memory_files(file_objs):
            if not file_objs:
                return "No files uploaded"

            saved_files = []
            for file_obj in file_objs:
                try:
                    if file_obj is None:
                        continue

                    # Get the filename and create destination path
                    filename = os.path.basename(file_obj.name)
                    dest_path = transcriber.config.memory_files_dir / filename

                    # Copy the file to memory_files
                    shutil.copy(file_obj.name, dest_path)
                    saved_files.append(filename)
                except Exception as e:
                    logger.error(f"Error saving file {file_obj.name if file_obj else 'None'}: {e}")

            if saved_files:
                return f"Saved {len(saved_files)} files: {', '.join(saved_files)}"
            else:
                return "No files were saved"

        # List all files in memory_files directory
        def list_memory_files():
            try:
                files = list(transcriber.config.memory_files_dir.glob("*.*"))
                if not files:
                    return "No files have been uploaded yet."

                file_list = "### Uploaded Files:\n\n"
                for i, file_path in enumerate(files, 1):
                    file_size = file_path.stat().st_size / 1024  # Size in KB
                    file_list += f"{i}. **{file_path.name}** ({file_size:.1f} KB)\n"

                return file_list
            except Exception as e:
                logger.error(f"Error listing memory files: {e}")
                return f"Error listing files: {str(e)}"

        # Connect the file upload buttons to their functions
        upload_memory_files_btn.click(
            fn=save_to_memory_files,
            inputs=[memory_files_upload],
            outputs=[memory_upload_status]
        ).then(
            fn=list_memory_files,
            inputs=[],
            outputs=[uploaded_files_display]
        )

        refresh_files_btn.click(
            fn=list_memory_files,
            inputs=[],
            outputs=[uploaded_files_display]
        )

        # Clear agent memory
        def clear_memory():
            result = transcriber.clear_agent_memory()
            return "Agent memory cleared."

        clear_memory_btn.click(
            fn=clear_memory,
            inputs=[],
            outputs=[memory_display]
        )

        # Update memory display
        def update_memory_display():
            if not hasattr(transcriber, 'agent_memory') or not transcriber.agent_memory:
                return "No memory stored."

            memory_text = "### Agent Memory:\n\n"
            for i, item in enumerate(reversed(transcriber.agent_memory), 1):
                memory_text += f"{i}. **{item['timestamp']}** - {item['transcription'][:50]}...\n\n"

            return memory_text

        # Copy functions
        def copy_to_clipboard(text):
            return None

        copy_analysis_btn.click(
            fn=copy_to_clipboard,
            inputs=[analysis_output],
            outputs=[],
            js="""
            async (text) => {
                if (!text) {
                    const el = document.getElementById('copy_analysis_btn');
                    const originalText = el.textContent;
                    el.textContent = 'Nothing to copy!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                    return [];
                }

                try {
                    await navigator.clipboard.writeText(text);
                    const el = document.getElementById('copy_analysis_btn');
                    const originalText = el.textContent;
                    el.textContent = '✓ Copied!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                } catch (err) {
                    console.error('Failed to copy: ', err);
                    const el = document.getElementById('copy_analysis_btn');
                    const originalText = el.textContent;
                    el.textContent = '❌ Copy failed!';
                    setTimeout(() => { el.textContent = originalText; }, 2000);
                }
                return [];
            }
            """
        )

    return app


# Launch the app
if __name__ == "__main__":
    try:
        # gradio
        logger.info("Gradio version: " + str(gr.__version__))

        import pyaudio

        logger.info("PyAudio version: " + str(pyaudio.__version__))
        import pydantic

        logger.info("Pydantic version: " + str(pydantic.__version__))
        import gspread

        logger.info("Gspread version: " + str(gspread.get_gspread_version()))
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Some dependencies are missing. Please install them to use all features.")
    app = create_interface()
    app.launch()