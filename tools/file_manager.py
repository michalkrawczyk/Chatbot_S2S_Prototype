import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
import json
import pandas as pd
import csv
from langchain_core.language_models.chat_models import BaseChatModel

from config import FILE_MEMORY_DIR, DATA_FILES_DIR, SUPPORTED_FILETYPES
from tools.tool_prompts_texts import file_summary_prompt
from tools.datasheet_manager import DATASHEET_MANAGER
from utils import conditional_logger_info


class FileInfo(BaseModel):
    """Model for file information stored in the index."""
    path: str
    description: str = ""
    origin: Optional[str] = None
    last_updated: Optional[str] = None


class FileSystemManager:
    """Class to manage file system operations."""
    _file_index = {}

    def __init__(self, memory_dir: str = FILE_MEMORY_DIR, memory_index_filename: str = "index.json"):
        self.memory_dir = memory_dir
        self.memory_index_path = os.path.join(DATA_FILES_DIR, memory_index_filename)

        # Ensure the memory directory exists
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
        Path(DATA_FILES_DIR).mkdir(parents=True, exist_ok=True)

        # Initialize or load the file index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load the file index."""
        if os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, 'r') as f:
                self._file_index = json.load(f)
        else:
            self._save_index()

    def _save_index(self):
        """Save the current file index to disk."""
        with open(self.memory_index_path, 'w') as f:
            json.dump(self._file_index, f, indent=2)

    def list_files(self) -> List[str]:
        """List all files in the memory directory."""
        files = os.listdir(self.memory_dir)
        conditional_logger_info(f"Files in memory directory ({self.memory_dir}): {files}")
        return [os.path.join(self.memory_dir, f) for f in files if os.path.isfile(os.path.join(self.memory_dir, f)) and f.endswith(SUPPORTED_FILETYPES)]

    def update_file_info(self, llm_summarizer: Optional[BaseChatModel] = None) -> "FileSystemManager":
        """Update file information using LLM for new or modified files.

        Args:
            llm_summarizer: LLM model to use for generating file summaries
        """
        if not llm_summarizer:
            raise ValueError("LLM summarizer model must be provided")

        current_files = self.list_files()

        # Check for new files
        for file_path in current_files:
            rel_path = os.path.relpath(file_path, self.memory_dir)

            if rel_path not in self._file_index:
                # New file found - generate description
                content = self.read_file(file_path)

                if content:
                    # Determine file type from extension
                    file_ext = os.path.splitext(file_path)[1].lower()
                    file_type = file_ext.lstrip('.') if file_ext else "text"

                    # Create the summary prompt
                    summary_prompt = file_summary_prompt(file_type, content)

                    # Generate summary using the LLM
                    from langchain.schema import HumanMessage
                    messages = [HumanMessage(content=summary_prompt)]
                    response = llm_summarizer.invoke(messages)
                    description = response.content

                    self._file_index[rel_path] = {
                        "path": file_path,
                        "description": description,
                        "origin": "undefined", #TODO
                        "last_updated": self._get_timestamp()
                    }
                else:
                    # If content is empty, just add a placeholder
                    self._file_index[rel_path] = {
                        "path": file_path,
                        "description": "No description available",
                        "origin": "undefined", #TODO
                        "last_updated": self._get_timestamp()
                    }

        # Remove entries for files that no longer exist
        existing_rel_paths = [os.path.relpath(p, self.memory_dir) for p in current_files]
        keys_to_remove = [k for k in self._file_index.keys() if k not in existing_rel_paths]

        for key in keys_to_remove:
            del self._file_index[key]

        self._save_index()
        return self

    def get_detailed_file_list(self, update_missing=False, llm_summarizer: Optional[BaseChatModel] = None):
        """Get a list of files with their descriptions.

        Args:
            update_missing: If True, update missing file information
            llm_summarizer: Function to summarize file content (required if update_missing is True)

        Returns:
            List of FileInfo objects
        """
        # TODO: reimplement?
        if update_missing and not llm_summarizer:
            raise ValueError("LLM summarizer must be provided when update_missing is True")

        current_files = self.list_files()
        result = []

        for file_path in current_files:
            rel_path = os.path.relpath(file_path, self.memory_dir)

            if rel_path in self._file_index:
                # File exists in index
                file_info = FileInfo(**self._file_index[rel_path])
                result.append(file_info)
            elif update_missing:
                # File not in index, but we want to update
                content = self.read_file(file_path)
                if content:
                    description = llm_summarizer(content)

                    file_info = FileInfo(
                        path=file_path,
                        description=description,
                        origin="system_scan",
                        last_updated=self._get_timestamp()
                    )

                    # Update the index
                    self._file_index[rel_path] = file_info.dict()
                    self._save_index()

                    result.append(file_info)
            else:
                # File not in index, but we don't want to update
                file_info = FileInfo(
                    path=file_path,
                    description="No description available"
                )
                result.append(file_info)

        return result

    def add_predefined_file_info(self, filename: str, description: str, origin: str = "user_defined",
                                 update_existing: bool = False):
        """Add or update predefined file information.

        Args:
            filename: Name of the file (can be relative to memory_dir or absolute)
            description: Description of the file
            origin: Origin of the file (default: user_defined)
            update_existing: Whether to update existing entries

        Returns:
            self for method chaining
        """
        # Normalize the file path
        if os.path.isabs(filename):
            file_path = filename
        else:
            file_path = os.path.join(self.memory_dir, filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        rel_path = os.path.relpath(file_path, self.memory_dir)

        # Check if the file is already in the index
        if rel_path in self._file_index:
            if update_existing:
                self._file_index[rel_path]["description"] = description
                self._file_index[rel_path]["origin"] = origin
                self._file_index[rel_path]["last_updated"] = self._get_timestamp()
        else:
            self._file_index[rel_path] = {
                "path": file_path,
                "description": description,
                "origin": origin,
                "last_updated": self._get_timestamp()
            }

        self._save_index()
        return self

    def read_file(self, file_path: str) -> str:
        """Read the content of a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            String representation of the file content
        """
        corrected_file_path = os.path.join(self.memory_dir, file_path) if not self.memory_dir in file_path else file_path

        if not os.path.exists(corrected_file_path):
            raise FileNotFoundError(f"File {corrected_file_path} does not exist")

        file_ext = os.path.splitext(corrected_file_path)[1].lower()


        try:
            if file_ext not in SUPPORTED_FILETYPES:
                raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {SUPPORTED_FILETYPES}")
            # Text files
            if file_ext in ['.txt', '.md']:
                with open(corrected_file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # CSV files
            elif file_ext == '.csv':
                DATASHEET_MANAGER.load_csv(corrected_file_path)
                return DATASHEET_MANAGER.df_as_str() # TODO: or should only give description?

            # Excel files
            elif file_ext in ['.xlsx', '.xls']:
                DATASHEET_MANAGER.load_excel(corrected_file_path)
                return DATASHEET_MANAGER.df_as_str() # TODO: or should only give description?

            # Word documents
            elif file_ext in ['.docx', '.doc']:
                raise NotImplementedError("Reading Word documents is not implemented yet.")
                # doc = docx.Document(file_path)
                # return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

            elif file_ext in ['.jpg', '.jpeg', '.png']:
                raise NotImplementedError("Reading images is not implemented yet.")

            # Default fallback - try to read as text
            else:
                try:
                    with open(corrected_file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    return f"[Binary file: {os.path.basename(corrected_file_path)}]"

        except Exception as e:
            return f"Error reading file {os.path.basename(corrected_file_path)}: {str(e)}"


    def _get_timestamp(self):
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()