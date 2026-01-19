"""
---
name: google_drive
version: 1.0.0
description: |
  Provides integration with Google Drive and Google Sheets, enabling the agent
  to download files from Google Drive and export Google Sheets as CSV files.
author: Chatbot Agent System
tags:
  - google
  - drive
  - sheets
  - cloud
  - download
  - integration
created_at: 2026-01-19
reference: https://code.claude.com/docs/en/skills
---

Google Drive Integration Skill

This skill provides Google Drive and Google Sheets integration capabilities:
- Download files from Google Drive
- Export Google Sheets as CSV files
- Support for specific sheet ranges in Google Sheets
- Automatic filename handling with uniqueness

The skill integrates with Google API services to provide seamless access to
cloud-stored files and spreadsheets.

Usage Example:
    from skills.google_drive_skill import GoogleDriveSkill
    
    skill = GoogleDriveSkill()
    tools = skill.tools  # Get the Google Drive tools
    
    # Use with agent
    agent.bind_tools(tools)

Requirements:
- Google API credentials must be configured
- Appropriate scopes for Drive and Sheets access

Reference: Claude Skills Methodology - External Service Integration
https://code.claude.com/docs/en/skills#external-services
"""

from typing import List, Any
from langchain_core.tools import tool

from skills.base_skill import BaseSkill, SkillMetadata
from tools.google_api_manager import GOOGLE_API_CLIENT, GoogleFileInput


class GoogleDriveSkill(BaseSkill):
    """
    Skill for Google Drive and Google Sheets integration.
    
    This skill provides tools for downloading files from Google Drive and
    exporting Google Sheets as CSV files. It handles both regular files
    and spreadsheets with a unified interface.
    
    Capabilities:
    - Download files from Google Drive by URL or file ID
    - Export Google Sheets as CSV with optional range selection
    - Automatic filename generation and uniqueness handling
    
    Authentication:
    - Requires Google API credentials to be configured
    - Uses service account or OAuth credentials
    - Fails gracefully if credentials not available
    
    Error Handling:
    - Invalid URLs are caught and reported
    - Authentication errors provide clear guidance
    - File download failures return descriptive messages
    
    Reference: Claude Skills Methodology - Cloud Integration
    https://code.claude.com/docs/en/skills#cloud-services
    """
    
    def __init__(self):
        """Initialize the Google Drive skill."""
        super().__init__()
    
    def _define_metadata(self) -> SkillMetadata:
        """Define metadata for the Google Drive skill."""
        return SkillMetadata(
            name="google_drive",
            version="1.0.0",
            description=(
                "Provides integration with Google Drive and Google Sheets for "
                "downloading files and exporting spreadsheets"
            ),
            author="Chatbot Agent System",
            tags=["google", "drive", "sheets", "cloud", "download", "integration"],
        )
    
    def _register_tools(self) -> List[Any]:
        """Register Google Drive tools."""
        
        @tool
        def download_google_file(params: GoogleFileInput) -> str:
            """
            Download a file from Google Drive or export a Google Sheet as CSV.
            
            This tool handles both Google Drive files and Google Sheets. For Google
            Sheets, it exports the data as a CSV file. The tool automatically uses
            the original filename if no output filename is specified and ensures
            unique filenames by appending a counter if needed.
            
            Args:
                params: GoogleFileInput containing:
                    - file_url: URL to the Google Drive file or Google Sheet
                    - output_filename: Optional custom filename for the download
                    - sheet_range: Optional cell range for Google Sheets (e.g., "A1:D10")
            
            Returns:
                Success message with file path, or error message
                
            Examples:
                Download a Google Drive file:
                >>> params = {
                ...     "file_url": "https://drive.google.com/file/d/FILE_ID/view"
                ... }
                >>> download_google_file(params)
                'Successfully downloaded Google Drive file to: /path/to/file.pdf'
                
                Export a Google Sheet with specific range:
                >>> params = {
                ...     "file_url": "https://docs.google.com/spreadsheets/d/SHEET_ID",
                ...     "sheet_range": "Sheet1!A1:D10",
                ...     "output_filename": "export.csv"
                ... }
                >>> download_google_file(params)
                'Successfully downloaded Google Sheet as CSV to: /path/to/export.csv'
            
            Error Handling:
                - Invalid URLs are caught and reported with helpful messages
                - Authentication failures provide guidance on credential setup
                - Network errors are logged and returned as error messages
                
            Authentication:
                Requires Google API credentials to be configured. If credentials
                are not available, returns an error message with setup instructions.
                
            Reference: Claude Skills Methodology - External API Integration
            https://code.claude.com/docs/en/skills#api-integration
            """
            try:
                if isinstance(params, dict):
                    params = GoogleFileInput(**params)
                
                # Determine if the URL is for a Google Sheet
                # Security: Use precise domain and path validation to prevent URL injection
                # Reference: Claude Skills Methodology - Input Validation
                # https://code.claude.com/docs/en/skills#security
                from urllib.parse import urlparse
                parsed_url = urlparse(params.file_url)
                
                # Check for Google Sheets domains and path patterns
                is_google_sheets_domain = parsed_url.netloc == "sheets.google.com" or parsed_url.netloc == "docs.google.com"
                has_spreadsheet_path = parsed_url.path.startswith("/spreadsheets/")
                is_sheet = is_google_sheets_domain and has_spreadsheet_path
                
                if is_sheet:
                    # Export Google Sheet as CSV
                    filepath = GOOGLE_API_CLIENT.save_sheet_to_csv(
                        spreadsheet_id_or_url=params.file_url,
                        output_file=params.output_filename,
                        sheet_range=params.sheet_range,
                    )
                    file_type = "Google Sheet as CSV"
                else:
                    # Download regular Google Drive file
                    filepath = GOOGLE_API_CLIENT.download_file(
                        file_id_or_url=params.file_url, 
                        output_file=params.output_filename
                    )
                    file_type = "Google Drive file"
                
                return f"Successfully downloaded {file_type} to: {filepath}"
                
            except Exception as e:
                return f"Error downloading file: {str(e)}"
        
        # Return list of tools
        return [download_google_file]
