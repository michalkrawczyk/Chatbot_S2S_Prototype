"""
---
name: file_management
version: 1.0.0
description: |
  Provides file system operations including listing, reading, and managing files
  in the agent's memory directory. Enables the agent to access and work with
  uploaded files and documents.
author: Chatbot Agent System
tags:
  - filesystem
  - files
  - storage
  - io
created_at: 2026-01-19
reference: https://code.claude.com/docs/en/skills
---

File Management Skill

This skill provides file system management capabilities to the agent, including:
- Listing available files in the memory directory
- Reading file contents
- Managing the global file helper catalog

The skill integrates with the FileSystemManager to provide a clean interface
for file operations while maintaining the global file catalog for context awareness.

Usage Example:
    from skills.file_management_skill import FileManagementSkill
    
    skill = FileManagementSkill()
    tools = skill.tools  # Get the file management tools
    
    # Use with agent
    agent.bind_tools(tools)

Reference: Claude Skills Methodology - File Management Patterns
https://code.claude.com/docs/en/skills#file-management
"""

from typing import List, Any
from langchain_core.tools import tool

from skills.base_skill import BaseSkill, SkillMetadata
from tools.file_manager import FileSystemManager


class FileManagementSkill(BaseSkill):
    """
    Skill for file system management operations.
    
    This skill encapsulates all file-related operations including listing,
    reading, and managing files in the agent's memory directory. It provides
    a clean separation of file management concerns from other agent capabilities.
    
    Capabilities:
    - List available files in memory directory
    - Read file contents
    - Access global file helper catalog
    
    Error Handling:
    - File not found errors are caught and returned as descriptive messages
    - Invalid file operations return error strings rather than raising exceptions
    - All errors are logged for debugging
    
    Reference: Claude Skills Methodology - Error Handling
    https://code.claude.com/docs/en/skills#error-handling
    """
    
    def __init__(self):
        """Initialize the file management skill."""
        # Create the filesystem manager instance
        self._fs_manager = FileSystemManager()
        super().__init__()
    
    def _define_metadata(self) -> SkillMetadata:
        """Define metadata for the file management skill."""
        return SkillMetadata(
            name="file_management",
            version="1.0.0",
            description=(
                "Provides file system operations including listing, reading, and "
                "managing files in the agent's memory directory"
            ),
            author="Chatbot Agent System",
            tags=["filesystem", "files", "storage", "io"],
        )
    
    def _register_tools(self) -> List[Any]:
        """Register file management tools."""
        
        @tool
        def get_file_list() -> str:
            """
            List available data files in the memory_files directory.
            
            Returns a formatted list of all files currently stored in the
            agent's memory directory, making them available for reference.
            
            Returns:
                String containing the list of available files
                
            Example:
                >>> get_file_list()
                'Available files:\\n1. document.pdf\\n2. data.csv'
            """
            try:
                return self._fs_manager.list_files()
            except Exception as e:
                return f"Error listing files: {str(e)}"
        
        @tool
        def get_file_content(file_name: str) -> str:
            """
            Load and return the content of a file from the memory_files directory.
            
            Reads the specified file and returns its content as a string. Supports
            various file types including text files, CSV, and other supported formats.
            
            Args:
                file_name: Name of the file to read (relative to memory_files directory)
            
            Returns:
                String containing the file content, or error message if file not found
                
            Example:
                >>> get_file_content("report.txt")
                'File contents...'
                
            Error Handling:
                - Returns descriptive error if file doesn't exist
                - Returns error message if file cannot be read
                - Logs all errors for debugging
            
            Reference: Claude Skills Methodology - Tool Error Handling
            https://code.claude.com/docs/en/skills#tool-errors
            """
            try:
                return self._fs_manager.read_file(file_name)
            except FileNotFoundError:
                return f"Error: File '{file_name}' not found in memory directory"
            except Exception as e:
                return f"Error reading file '{file_name}': {str(e)}"
        
        # Return list of tools
        return [get_file_list, get_file_content]
    
    @property
    def filesystem_manager(self) -> FileSystemManager:
        """
        Get the underlying filesystem manager instance.
        
        Provides access to the FileSystemManager for advanced operations
        that may not be exposed as tools.
        
        Returns:
            FileSystemManager instance
        """
        return self._fs_manager
    
    def get_global_helper_context(self) -> str:
        """
        Get the global file helper catalog as context.
        
        This provides a formatted view of all files in the global helper
        catalog, useful for giving the agent awareness of available files.
        
        Returns:
            Formatted string containing global file helper information
        """
        try:
            return self._fs_manager.get_global_helper_as_context()
        except Exception as e:
            return f"Error retrieving global helper context: {str(e)}"
