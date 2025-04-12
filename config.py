import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_MEMORY_DIR = os.path.join(MAIN_DIR, "memory_files")
DATA_FILES_DIR = os.path.join(MAIN_DIR, "data_files")
SUPPORTED_FILETYPES = (".txt", ".pdf", ".docx", ".jpg", ".png", ".csv", ".xls", ".xlsx")


# Settings
ADDITIONAL_LOGGER_INFO = True
RECURSION_LIMIT = 10
AGENT_TRACE = True
AGENT_VERBOSE = True
KEEP_LAST_UPLOADED_FILE_IN_CONTEXT = True