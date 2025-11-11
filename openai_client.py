import os
import tempfile
import time
import uuid
from pathlib import Path

from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

from general.logs import logger
from stt_utils import SUPPORT_LANGUAGES_CAST_DICT, SUPPORT_LANGUAGES


class OpenAIClient:
    """Handles OpenAI API interactions - Pure API client for connection and low-level API calls"""

    def __init__(self, default_api_key=""):
        self.client = None
        self.connected = False
        self.last_error = None

        # Don't store API key in memory - only use it to initialize client
        # Try to connect with default key
        if default_api_key:
            self.connect(default_api_key)

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
            client = OpenAI(api_key=api_key)
            # Test connection with a lightweight call
            client.models.list()
            self.client = client
            self.connected = True
            self.last_error = None
            logger.info("Successfully connected to OpenAI API")
            return "✓ Successfully connected to OpenAI API", True, "green"
        except AuthenticationError as e:
            error_msg = "Invalid API key"
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI authentication error: {str(e)}")
            return f"❌ Authentication failed: {error_msg}", False, "red"
        except (RateLimitError, APIConnectionError) as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI connection error: {error_msg}")
            return f"❌ Error connecting to OpenAI: {error_msg}", False, "red"
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.connected = False
            logger.error(f"OpenAI connection error: {error_msg}")
            return f"❌ Error connecting to OpenAI: {error_msg}", False, "red"

    def text_to_speech(self, text, voice="alloy", stream=True):
        """
        Convert text to speech using OpenAI's TTS API with streaming support

        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use
            stream (bool): Whether to use streaming mode

        Returns:
            bytes or streaming_response or str: Audio content, streaming response, or file path
        """
        if not self.client:
            return "Error: Not connected to OpenAI API"

        try:
            if stream:
                # Use the streaming response approach
                response = self.client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    instructions="Speak with soft, calm voice and a conversational manner.",
                    voice=voice,
                    input=text,
                    response_format="wav",
                )
                return response
            else:
                # Non-streaming approach for saving to file
                response = self.client.audio.speech.create(
                    model="tts-1", voice=voice, input=text
                )

                # Save to file with unique filename to avoid race conditions
                temp_dir = Path(tempfile.gettempdir()) / "spaces_audio"
                temp_dir.mkdir(exist_ok=True, parents=True)
                # Use UUID for guaranteed uniqueness instead of timestamp
                unique_id = uuid.uuid4().hex[:8]
                output_path = str(temp_dir / f"tts_{unique_id}_{int(time.time())}.mp3")
                response.stream_to_file(output_path)
                logger.info(f"Text-to-speech audio saved to: {output_path}")
                return output_path

        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return f"Error generating speech: {str(e)}"
