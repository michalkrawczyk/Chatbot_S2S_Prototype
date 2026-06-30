import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_MEMORY_DIR = os.path.join(MAIN_DIR, "memory_files")
DATA_FILES_DIR = os.path.join(MAIN_DIR, "data_files")
SUPPORTED_FILETYPES = (".txt", ".pdf", ".docx", ".jpg", ".png", ".csv", ".xls", ".xlsx")


# Settings
ADDITIONAL_LOGGER_INFO = True
RECURSION_LIMIT = 20
AGENT_TRACE = True
AGENT_VERBOSE = True
KEEP_LAST_UPLOADED_FILE_IN_CONTEXT = True

# Speech-to-Text Model Configuration
DEFAULT_STT_MODEL = "whisper"  # Options: "whisper", "nemo"
SUPPORTED_STT_MODELS = ["whisper", "nemo"]

# LLM Model Configuration
OPENAI_MODELS = ["o3-mini", "gpt-4-turbo", "gpt-4o", "gpt-4.1-mini", "gpt-5-mini"]
# Ollama runs GGUF models via llama.cpp under the hood — no API key required
OLLAMA_MODELS = ["lfm2.5:350m", "lfm2.5:1.2b-instruct", "lfm2.5:8b-a1b"]
