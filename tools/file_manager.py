import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
import pandas as pd
from pydantic import BaseModel

from general.config import DATA_FILES_DIR, FILE_MEMORY_DIR, SUPPORTED_FILETYPES
from general.logs import conditional_logger_info, logger
from tools.datasheet_manager import DATASHEET_MANAGER
from tools.tool_prompts_texts import file_summary_prompt


# Constants for description placeholders
PLACEHOLDER_GENERATING = "Generating description..."
PLACEHOLDER_NO_DESCRIPTION = "No description available"
PLACEHOLDER_ERROR = "Error generating description"


class FileInfo(BaseModel):
    """Model for file information stored in the index."""

    file: Optional[str] = None  # Filename only (optional for backward compatibility)
    path: str  # Full absolute path
    description: str = ""
    origin: Optional[str] = None
    last_updated: Optional[str] = None


class FileSystemManager:
    """Class to manage file system operations with a global file helper catalog."""

    def __init__(
        self,
        memory_dir: str = FILE_MEMORY_DIR,
        memory_index_filename: str = "index.json",
        global_helper_filename: str = "global_file_helper.json",
    ):
        self.memory_dir = memory_dir
        self.memory_index_path = os.path.join(DATA_FILES_DIR, memory_index_filename)
        self.global_helper_path = os.path.join(DATA_FILES_DIR, global_helper_filename)

        # Per-instance indexes
        self._file_index = {}
        self._global_helper_index = {}
        # Store reference to LLM for async operations (set by set_llm_summarizer)
        self._llm_summarizer = None
        # Thread pool for async description updates (limit concurrent operations)
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="desc_updater")
        # Ensure the memory directory exists
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
        Path(DATA_FILES_DIR).mkdir(parents=True, exist_ok=True)

        # Initialize or load the file index
        self._initialize_index()
        self._initialize_global_helper()

    def _initialize_index(self):
        """Initialize or load the file index."""
        if os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, "r") as f:
                self._file_index = json.load(f)
        else:
            self._save_index()

    def _initialize_global_helper(self):
        """Initialize or load the global file helper.
        
        After loading, triggers an async operation to update any entries
        with missing or empty descriptions.
        """
        if os.path.exists(self.global_helper_path):
            try:
                with open(self.global_helper_path, "r") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(
                    "Failed to load global helper file '%s': %s. "
                    "Reinitializing with an empty catalog.",
                    self.global_helper_path,
                    exc,
                )
                self._global_helper_index = {}
                self._save_global_helper()
                return

            # Convert from list format to dict format for easier lookup
            if isinstance(data, list):
                helper_index = {}
                for item in data:
                    # Validate each entry before using the "path" key
                    if isinstance(item, dict) and "path" in item:
                        helper_index[item["path"]] = item
                    else:
                        logger.warning(
                            "Skipping invalid global helper entry without 'path': %s",
                            item,
                        )
                self._global_helper_index = helper_index
            else:
                self._global_helper_index = data
        else:
            self._save_global_helper()
        
        # Trigger async update for missing descriptions
        self._trigger_async_description_update()

    def _save_index(self):
        """Save the current file index to disk."""
        with open(self.memory_index_path, "w") as f:
            json.dump(self._file_index, f, indent=2)

    def _save_global_helper(self):
        """Save the global helper index to disk as a JSON array."""
        # Convert dict to list format for output
        helper_list = list(self._global_helper_index.values())
        with open(self.global_helper_path, "w") as f:
            json.dump(helper_list, f, indent=2)

    def list_files(self) -> List[str]:
        """List all files in the memory directory."""
        files = os.listdir(self.memory_dir)
        conditional_logger_info(
            f"Files in memory directory ({self.memory_dir}): {files}"
        )
        return [
            os.path.join(self.memory_dir, f)
            for f in files
            if os.path.isfile(os.path.join(self.memory_dir, f))
            and f.endswith(SUPPORTED_FILETYPES)
        ]

    def update_file_info(
        self, llm_summarizer: Optional[BaseChatModel] = None
    ) -> "FileSystemManager":
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
                    file_type = file_ext.lstrip(".") if file_ext else "text"

                    # Create the summary prompt
                    summary_prompt = file_summary_prompt(file_type, content)

                    # Generate summary using the LLM
                    messages = [HumanMessage(content=summary_prompt)]
                    response = llm_summarizer.invoke(messages)
                    description = response.content

                    self._file_index[rel_path] = {
                        "path": file_path,
                        "description": description,
                        "origin": "undefined",  # TODO
                        "last_updated": self._get_timestamp(),
                    }
                    # Also add to global helper
                    self._update_global_helper_entry(
                        file_path=file_path,
                        description=description,
                        origin="file_scan",
                    )
                else:
                    # If content is empty, just add a placeholder
                    self._file_index[rel_path] = {
                        "path": file_path,
                        "description": PLACEHOLDER_NO_DESCRIPTION,
                        "origin": "undefined",  # TODO
                        "last_updated": self._get_timestamp(),
                    }
                    # Also add to global helper
                    self._update_global_helper_entry(
                        file_path=file_path,
                        description=PLACEHOLDER_NO_DESCRIPTION,
                        origin="file_scan",
                    )

        # Remove entries for files that no longer exist
        existing_rel_paths = [
            os.path.relpath(p, self.memory_dir) for p in current_files
        ]
        keys_to_remove = [
            k for k in self._file_index.keys() if k not in existing_rel_paths
        ]

        for key in keys_to_remove:
            del self._file_index[key]

        self._save_index()
        return self

    def get_detailed_file_list(
        self, update_missing=False, llm_summarizer: Optional[BaseChatModel] = None
    ):
        """Get a list of files with their descriptions.

        Args:
            update_missing: If True, update missing file information
            llm_summarizer: Function to summarize file content (required if update_missing is True)

        Returns:
            List of FileInfo objects
        """
        # TODO: reimplement?
        if update_missing and not llm_summarizer:
            raise ValueError(
                "LLM summarizer must be provided when update_missing is True"
            )

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
                        last_updated=self._get_timestamp(),
                    )

                    # Update the index
                    self._file_index[rel_path] = file_info.dict()
                    self._save_index()

                    result.append(file_info)
            else:
                # File not in index, but we don't want to update
                file_info = FileInfo(
                    path=file_path, description=PLACEHOLDER_NO_DESCRIPTION
                )
                result.append(file_info)

        return result

    def add_predefined_file_info(
        self,
        filename: str,
        description: str,
        origin: str = "user_defined",
        update_existing: bool = False,
    ):
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
                "last_updated": self._get_timestamp(),
            }

        self._save_index()
        return self

    def read_file(self, file_path: str) -> str:
        """Read the content of a file based on its extension.
        
        This method is kept for backward compatibility. For structured results,
        use read_file_safe() instead.

        Args:
            file_path: Path to the file

        Returns:
            String representation of the file content, or error message
        """
        success, content = self.read_file_safe(file_path)
        return content
    
    def read_file_safe(self, file_path: str) -> tuple[bool, str]:
        """Read the content of a file and return structured result.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (success: bool, content: str)
            - If successful: (True, file content as string)
            - If error: (False, error message)
        """
        corrected_file_path = (
            os.path.join(self.memory_dir, file_path)
            if not self.memory_dir in file_path
            else file_path
        )

        if not os.path.exists(corrected_file_path):
            return (False, f"File {corrected_file_path} does not exist")

        file_ext = os.path.splitext(corrected_file_path)[1].lower()

        try:
            if file_ext not in SUPPORTED_FILETYPES:
                return (False, f"Unsupported file format: {file_ext}. Supported formats: {SUPPORTED_FILETYPES}")
            
            # Text files
            if file_ext in [".txt", ".md"]:
                with open(corrected_file_path, "r", encoding="utf-8") as f:
                    return (True, f.read())

            # CSV files
            elif file_ext == ".csv":
                DATASHEET_MANAGER.load_csv(corrected_file_path)
                return (True, DATASHEET_MANAGER.df_as_str())

            # Excel files
            elif file_ext in [".xlsx", ".xls"]:
                DATASHEET_MANAGER.load_excel(corrected_file_path)
                return (True, DATASHEET_MANAGER.df_as_str())

            # Word documents
            elif file_ext in [".docx", ".doc"]:
                return (False, "Reading Word documents is not implemented yet.")

            elif file_ext in [".jpg", ".jpeg", ".png"]:
                return (False, "Reading images is not implemented yet.")

            # Default fallback - try to read as text
            else:
                try:
                    with open(corrected_file_path, "r", encoding="utf-8") as f:
                        return (True, f.read())
                except UnicodeDecodeError:
                    return (False, f"[Binary file: {os.path.basename(corrected_file_path)}]")

        except Exception as e:
            return (False, f"Error reading file {os.path.basename(corrected_file_path)}: {str(e)}")

    def _get_timestamp(self):
        """Get the current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _update_global_helper_entry(
        self,
        file_path: str,
        description: str = "",
        origin: str = "user_defined",
        update_existing: bool = True,
    ):
        """Update or add an entry to the global helper catalog.
        
        Args:
            file_path: Full absolute path to the file
            description: Description of the file's purpose and content
            origin: Origin of the file information
            update_existing: Whether to update existing entries
        """
        # Normalize path to absolute
        abs_path = os.path.abspath(file_path)
        filename = os.path.basename(abs_path)
        
        # Check if entry exists
        if abs_path in self._global_helper_index:
            if update_existing:
                self._global_helper_index[abs_path]["description"] = description
                self._global_helper_index[abs_path]["origin"] = origin
                self._global_helper_index[abs_path]["last_updated"] = self._get_timestamp()
        else:
            # Create new entry
            self._global_helper_index[abs_path] = {
                "file": filename,
                "path": abs_path,
                "description": description,
                "origin": origin,
                "last_updated": self._get_timestamp(),
            }
        
        self._save_global_helper()

    def add_file_to_global_helper(
        self,
        file_path: str,
        description: str = "",
        origin: str = "user_upload",
        llm_summarizer: Optional[BaseChatModel] = None,
    ):
        """Add a file to the global helper catalog with optional LLM-generated description.
        
        This method runs asynchronously: if no description is provided and an LLM
        summarizer is supplied, it will start description generation in a background thread.
        
        Args:
            file_path: Full path to the file
            description: Optional pre-defined description
            origin: Origin of the file
            llm_summarizer: Optional LLM to generate description if not provided
            
        Returns:
            self for method chaining
        """
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File {abs_path} does not exist")
        
        # If no description provided and LLM is available, generate one asynchronously
        if not description and llm_summarizer:
            # Add entry immediately with placeholder
            self._update_global_helper_entry(
                file_path=abs_path,
                description=PLACEHOLDER_GENERATING,
                origin=origin,
            )
            
            # Start async description generation in thread pool
            def generate_description_async():
                try:
                    success, content = self.read_file_safe(abs_path)
                    # If we have valid content, attempt to generate a summary
                    if success and content:
                        # Determine file type
                        file_ext = os.path.splitext(abs_path)[1].lower()
                        file_type = file_ext.lstrip(".") if file_ext else "text"
                        
                        # Generate summary
                        summary_prompt = file_summary_prompt(file_type, content)
                        messages = [HumanMessage(content=summary_prompt)]
                        response = llm_summarizer.invoke(messages)
                        generated_description = response.content
                    else:
                        # Handle file read errors
                        generated_description = self._handle_description_error(success, content)
                except Exception as e:
                    logger.error(f"Error generating description for {abs_path}: {str(e)}")
                    generated_description = PLACEHOLDER_ERROR
                
                # Update with the generated description
                self._update_global_helper_entry(
                    file_path=abs_path,
                    description=generated_description,
                    origin=origin,
                    update_existing=True,
                )
                logger.info(f"Async description generated for {os.path.basename(abs_path)}")
            
            # Submit to thread pool executor
            self._executor.submit(generate_description_async)
        elif not description:
            # No LLM available, add with placeholder
            self._update_global_helper_entry(
                file_path=abs_path,
                description=PLACEHOLDER_NO_DESCRIPTION,
                origin=origin,
            )
        else:
            # Description provided, add it directly
            self._update_global_helper_entry(
                file_path=abs_path,
                description=description,
                origin=origin,
            )
        
        return self

    def get_global_helper_as_context(self) -> str:
        """Get the global file helper as a formatted string for agent context.
        
        Returns:
            Formatted string representation of the global helper
        """
        if not self._global_helper_index:
            return "No files in global helper catalog."
        
        context = "### Global File Helper Catalog:\n\n"
        for file_info in self._global_helper_index.values():
            file_display_name = file_info.get("file", os.path.basename(file_info["path"]))
            context += f"**{file_display_name}**\n"
            context += f"  - Description: {file_info['description']}\n\n"
        
        return context

    def get_file_description_from_helper(self, file_path: str) -> Optional[str]:
        """Get a file's description from the global helper.
        
        Args:
            file_path: Path to the file (can be relative or absolute)
            
        Returns:
            Description string if found, None otherwise
        """
        abs_path = os.path.abspath(file_path)
        
        if abs_path in self._global_helper_index:
            return self._global_helper_index[abs_path].get("description", None)
        
        return None
    
    def cleanup_deleted_files(self):
        """Remove entries for files that no longer exist from the global helper.
        
        This method scans all entries in the global helper and removes those
        where the file no longer exists on the filesystem.
        
        Returns:
            Number of entries removed
        """
        paths_to_remove = []
        
        for file_path in self._global_helper_index.keys():
            if not os.path.exists(file_path):
                paths_to_remove.append(file_path)
        
        for path in paths_to_remove:
            logger.info(f"Removing deleted file from global helper: {os.path.basename(path)}")
            del self._global_helper_index[path]
        
        if paths_to_remove:
            self._save_global_helper()
        
        return len(paths_to_remove)
    
    def cleanup(self):
        """Cleanup resources used by the FileSystemManager.
        
        This method should be called when the FileSystemManager instance
        is no longer needed to ensure proper cleanup of the thread pool executor.
        Waits for all pending tasks to complete before shutting down.
        """
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("FileSystemManager thread pool executor shutdown")
    
    def set_llm_summarizer(self, llm_summarizer: Optional[BaseChatModel]):
        """Set the LLM summarizer for async description generation.
        
        This should be called after the LLM is initialized to enable
        automatic description updates for files with missing descriptions.
        
        Args:
            llm_summarizer: LLM model to use for generating file summaries
            
        Returns:
            self for method chaining
        """
        self._llm_summarizer = llm_summarizer
        logger.info("LLM summarizer set for FileSystemManager")
        
        # Trigger async update if there are files with missing descriptions
        self._trigger_async_description_update()
        
        return self
    
    def _trigger_async_description_update(self):
        """Trigger async update for files with missing or empty descriptions.
        
        This method scans the global helper index for entries with missing
        descriptions and starts background threads to generate them using
        the configured LLM summarizer. This is non-blocking and runs
        asynchronously.
        """
        # Only proceed if we have an LLM summarizer configured
        if not self._llm_summarizer:
            logger.debug("No LLM summarizer configured, skipping async description update")
            return
        
        # Find files with missing descriptions
        files_to_update = []
        for file_path, file_info in self._global_helper_index.items():
            description = file_info.get("description", "")
            # Check if description is missing, empty, or a placeholder
            if (not description or 
                description in [PLACEHOLDER_NO_DESCRIPTION, PLACEHOLDER_GENERATING, PLACEHOLDER_ERROR]):
                # Verify file still exists before attempting update
                if os.path.exists(file_path):
                    files_to_update.append((file_path, file_info.get("origin", "unknown")))
        
        if not files_to_update:
            logger.debug("No files with missing descriptions found")
            return
        
        logger.info(f"Found {len(files_to_update)} file(s) with missing descriptions, starting async update")
        
        # Start async update for each file
        for file_path, origin in files_to_update:
            self._update_description_async(file_path, origin)
    
    def _handle_description_error(self, success: bool, content: str) -> str:
        """Handle errors when reading file content for description generation.
        
        Args:
            success: Whether the file read was successful
            content: The file content (if success) or error message (if not success)
            
        Returns:
            Appropriate description based on the error condition
        """
        if not success:
            # File read failed, content contains the error message
            return content
        else:
            # Success but no content
            return PLACEHOLDER_NO_DESCRIPTION
    
    def _update_description_async(self, file_path: str, origin: str):
        """Update a file's description asynchronously in a background thread.
        
        Uses the thread pool executor to manage concurrent operations and
        prevent resource exhaustion.
        
        Args:
            file_path: Full path to the file
            origin: Origin of the file entry
        """
        def generate_and_update():
            try:
                filename = os.path.basename(file_path)
                logger.info(f"Starting async description generation for {filename}")
                
                # Mark as generating
                self._update_global_helper_entry(
                    file_path=file_path,
                    description=PLACEHOLDER_GENERATING,
                    origin=origin,
                    update_existing=True,
                )
                
                # Read file content
                success, content = self.read_file_safe(file_path)
                
                if success and content:
                    # Determine file type
                    file_ext = os.path.splitext(file_path)[1].lower()
                    file_type = file_ext.lstrip(".") if file_ext else "text"
                    
                    # Generate summary using LLM
                    summary_prompt = file_summary_prompt(file_type, content)
                    messages = [HumanMessage(content=summary_prompt)]
                    response = self._llm_summarizer.invoke(messages)
                    generated_description = response.content
                    
                    logger.info(f"Successfully generated description for {filename}")
                else:
                    # Handle file read errors
                    generated_description = self._handle_description_error(success, content)
                    logger.warning(f"Could not read file content for {filename}: {generated_description}")
                    
            except Exception as e:
                logger.error(f"Error generating async description for {filename}: {str(e)}")
                generated_description = PLACEHOLDER_ERROR
            
            # Update with the generated description
            self._update_global_helper_entry(
                file_path=file_path,
                description=generated_description,
                origin=origin,
                update_existing=True,
            )
            logger.info(f"Completed async description update for {filename}")
        
        # Submit to thread pool executor
        self._executor.submit(generate_and_update)
