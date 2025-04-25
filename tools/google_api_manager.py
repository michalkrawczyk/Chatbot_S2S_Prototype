import os
import json
import io
import csv
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pydantic import BaseModel, Field
from typing import Optional

from general.logs import logger
from general.config import FILE_MEMORY_DIR


class GoogleClient:
    """Client for interacting with Google Sheets and Google Drive."""

    def __init__(self, env_var_name="GOOGLE_SERVICE_KEY_JSON", memory_dir: str = FILE_MEMORY_DIR):
        """
        Initialize with the environment variable containing the service key JSON.

        Args:
            env_var_name (str): Name of the environment variable holding the service key JSON.
            memory_dir (str): Directory to save downloaded files.

        Raises:
            ValueError: If the environment variable is not set or contains invalid JSON.
        """
        service_key_json = os.environ.get(env_var_name)
        self.memory_dir = memory_dir

        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)

        if not service_key_json:
            raise ValueError(f"Environment variable {env_var_name} not set")

        try:
            self.service_key_dict = json.loads(service_key_json)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {env_var_name}")

        # Get credentials
        self.credentials = service_account.Credentials.from_service_account_info(
            self.service_key_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive']
        )

        # Initialize services
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.sheets_service = build('sheets', 'v4', credentials=self.credentials)

    def get_sheet_names(self, spreadsheet_id_or_url):
        """
        Get all sheet names from a Google Spreadsheet.

        Args:
            spreadsheet_id_or_url (str): The ID or URL of the spreadsheet.

        Returns:
            list: List of sheet names.
        """
        spreadsheet_id = self.extract_file_id(spreadsheet_id_or_url)

        # Get spreadsheet metadata including sheets
        spreadsheet = self.sheets_service.spreadsheets().get(
            spreadsheetId=spreadsheet_id
        ).execute()

        # Extract sheet names
        sheets = spreadsheet.get('sheets', [])
        sheet_names = [sheet.get('properties', {}).get('title', '') for sheet in sheets]

        return sheet_names

    def extract_file_id(self, url):
        """
        Extract file ID from Google Drive or Sheets URL.

        Args:
            url (str): Google Drive or Sheets URL.

        Returns:
            str: The file ID.

        Raises:
            ValueError: If no valid file ID could be extracted.
        """
        # Patterns for different Google URL formats
        patterns = [
            r'/d/([a-zA-Z0-9-_]+)',  # Drive: /d/{fileId}
            r'/file/d/([a-zA-Z0-9-_]+)',  # Drive: /file/d/{fileId}
            r'id=([a-zA-Z0-9-_]+)',  # Drive: ?id={fileId}
            r'/spreadsheets/d/([a-zA-Z0-9-_]+)',  # Sheets: /spreadsheets/d/{fileId}
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # If URL is just the ID itself
        if re.match(r'^[a-zA-Z0-9-_]{25,}$', url):
            return url

        raise ValueError(f"Could not extract file ID from URL: {url}")

    def get_file_metadata(self, file_id_or_url):
        """
        Get metadata for a file from Google Drive.

        Args:
            file_id_or_url (str): ID or URL of the file.

        Returns:
            dict: The file metadata.
        """
        file_id = self.extract_file_id(file_id_or_url)
        return self.drive_service.files().get(fileId=file_id, fields="name,mimeType").execute()

    def _get_unique_filepath(self, filepath):
        """
        Generate a unique filepath if the given path already exists.

        Args:
            filepath (str): The original file path.

        Returns:
            str: A unique file path.
        """
        if not os.path.exists(filepath):
            return filepath

        # Split the path into directory, filename and extension
        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)

        counter = 1
        while True:
            new_path = os.path.join(directory, f"{name}_{counter}{ext}")
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def _get_full_path(self, filename):
        """
        Get the full path for a file in the memory directory.

        Args:
            filename (str): The filename or path.

        Returns:
            str: The full path in the memory directory.
        """
        # If the path is already absolute and outside memory_dir, move it to memory_dir
        if os.path.isabs(filename) and not filename.startswith(self.memory_dir):
            return os.path.join(self.memory_dir, os.path.basename(filename))

        # If it's a relative path, ensure it's within memory_dir
        if not filename.startswith(self.memory_dir):
            return os.path.join(self.memory_dir, filename)

        return filename

    # Google Sheets Methods

    def get_sheet_values(self, spreadsheet_id_or_url, sheet_range=None):
        """
        Get values from a spreadsheet.

        Args:
            spreadsheet_id_or_url (str): The ID or URL of the spreadsheet.
            sheet_range (str, optional): Range of the worksheet (e.g., "Sheet1", "Sheet1!A1:D10").
                                        If None, returns data from all sheets.

        Returns:
            dict or list: If sheet_range is None, returns a dictionary mapping sheet names to their values.
                         Otherwise, returns a list of rows for the specified sheet range.
        """
        spreadsheet_id = self.extract_file_id(spreadsheet_id_or_url)

        # If no sheet range is specified, get all sheets
        if sheet_range is None:
            sheet_names = self.get_sheet_names(spreadsheet_id)
            result = {}
            for sheet_name in sheet_names:
                sheet_data = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_name
                ).execute()
                result[sheet_name] = sheet_data.get('values', [])
            return result
        else:
            # Original behavior for specific sheet range
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=sheet_range
            ).execute()
            return result.get('values', [])

    def save_sheet_to_csv(self, spreadsheet_id_or_url,
                          output_file:Optional[str] = None,
                          sheet_range:Optional[str] = None):
        """
        Save a spreadsheet to a CSV file.

        Args:
            spreadsheet_id_or_url (str): The ID or URL of the spreadsheet.
            output_file (str, optional): Path to the output CSV file. If None, uses the original filename.
            sheet_range (str, optional): Range of the worksheet. If None, saves all sheets to separate CSV files.

        Returns:
            str or list: Path to the saved CSV file, or list of paths if multiple sheets were saved.
        """
        spreadsheet_id = self.extract_file_id(spreadsheet_id_or_url)

        # Get original filename if output_file is None
        if output_file is None:
            metadata = self.get_file_metadata(spreadsheet_id)
            base_filename = metadata['name']
        else:
            base_filename = os.path.splitext(os.path.basename(output_file))[0]

        # If no sheet range is specified, save all sheets
        if sheet_range is None:
            sheet_names = self.get_sheet_names(spreadsheet_id)
            saved_files = []

            for sheet_name in sheet_names:
                values = self.get_sheet_values(spreadsheet_id_or_url, sheet_name)

                # Create filename for this sheet
                sheet_filename = f"{base_filename}_{sheet_name}.csv"
                full_path = self._get_full_path(sheet_filename)

                # Check if the directory exists
                directory = os.path.dirname(full_path)
                if not os.path.isdir(directory):
                    logger.warning(f"Directory {directory} does not exist. Saving file at {self.memory_dir}")
                    full_path = os.path.join(self.memory_dir, os.path.basename(sheet_filename))

                # Ensure the file path is unique
                full_path = self._get_unique_filepath(full_path)

                with open(full_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(values)

                logger.info(f"Saved sheet '{sheet_name}' to {full_path}")
                saved_files.append(full_path)

            return saved_files
        else:
            # Original behavior for specific sheet range
            values = self.get_sheet_values(spreadsheet_id_or_url, sheet_range)

            # Append .csv if not already in the filename
            if output_file is None:
                output_file = f"{base_filename}.csv"
            elif not output_file.lower().endswith('.csv'):
                output_file = f"{output_file}.csv"

            full_path = self._get_full_path(output_file)

            # Check if the directory exists
            directory = os.path.dirname(full_path)
            if not os.path.isdir(directory):
                logger.warning(f"Directory {directory} does not exist. Saving file at {self.memory_dir}")
                full_path = os.path.join(self.memory_dir, os.path.basename(output_file))

            # Ensure the file path is unique
            full_path = self._get_unique_filepath(full_path)

            with open(full_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(values)

            logger.info(f"Saved sheet to {full_path}")
            return full_path

    # Google Drive Methods

    def list_files(self, max_results=10, query=None):
        """
        List files from Google Drive with optional filtering.

        Args:
            max_results (int, optional): Maximum number of files to return.
            query (str, optional): Query string for filtering files.

        Returns:
            list: List of file metadata.
        """
        results = self.drive_service.files().list(
            pageSize=max_results,
            fields="nextPageToken, files(id, name, mimeType)",
            q=query
        ).execute()
        return results.get('files', [])

    def download_file(self, file_id_or_url, output_file=None):
        """
        Download a file from Google Drive.

        Args:
            file_id_or_url (str): ID or URL of the file to download.
            output_file (str, optional): Path to the output file. If None, uses the original filename.

        Returns:
            str: The full path to the downloaded file.
        """
        file_id = self.extract_file_id(file_id_or_url)

        # Get original filename if output_file is None
        if output_file is None:
            metadata = self.get_file_metadata(file_id)
            output_file = metadata['name']

        request = self.drive_service.files().get_media(fileId=file_id)

        # Get the full path in memory directory
        full_path = self._get_full_path(output_file)

        # Check if the directory exists
        directory = os.path.dirname(full_path)
        if not os.path.isdir(directory):
            logger.warning(f"Directory {directory} does not exist. Saving file at {self.memory_dir}")
            full_path = os.path.join(self.memory_dir, os.path.basename(output_file))

        # Ensure the file path is unique
        full_path = self._get_unique_filepath(full_path)

        with open(full_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}%")

        logger.info(f"Downloaded file to {full_path}")
        return full_path

    def download_file_by_name(self, file_name, output_file=None, exact_match=True):
        """
        Download a file from Google Drive by its name.

        Args:
            file_name (str): Name of the file to download.
            output_file (str, optional): Path to the output file. If None, uses the original filename.
            exact_match (bool, optional): If True, searches for exact name match.

        Returns:
            str or bool: The full path to the downloaded file if successful, False otherwise.
        """
        if exact_match:
            query = f"name = '{file_name}'"
        else:
            query = f"name contains '{file_name}'"

        files = self.list_files(max_results=1, query=query)
        if not files:
            logger.warning(f"No file found with name: {file_name}")
            return False

        # Use original filename if output_file is None
        if output_file is None:
            output_file = files[0]['name']

        return self.download_file(files[0]['id'], output_file)





class GoogleFileInput(BaseModel):
    """Input for downloading a file from Google Drive or Google Sheets."""

    file_url: str = Field(
        ...,
        description="URL or file ID of the Google Drive file or Google Sheet to download"
    )

    output_filename: Optional[str] = Field(
        None,
        description="Name for the downloaded file. If None, uses the original filename"
    )

    sheet_range: Optional[str] = Field(
        None,
        description="For Google Sheets only: range of cells to export (e.g., 'Sheet1!A1:D10'). If None, all sheets will be retrieved"
    )

### Singleton instance
GOOGLE_API_CLIENT = GoogleClient()