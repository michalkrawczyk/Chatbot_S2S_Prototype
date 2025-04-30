from typing import Union, Optional, List, Dict, Any
from langchain.tools import Tool
from langchain_core.tools import tool

from tools.file_manager import FileSystemManager
from tools.datasheet_manager import (
    DATASHEET_MANAGER,
    DatasheetLoadParams,
    DatasheetChunkParams,
    DatasheetStatsReqParams,
)
from tools.google_api_manager import GOOGLE_API_CLIENT, GoogleFileInput

import traceback
import logging

logger = logging.getLogger("Tool Register")


#
# Create the filesystem tool instance
FILESYSTEM_MANAGER = FileSystemManager()


def _read_datasheet(file_path: str, sheet_name: Optional[Union[str, int]] = None):
    """
    Load a datasheet file and return its content as a string.
    """
    if file_path.endswith((".csv", ".CSV")):
        DATASHEET_MANAGER.load_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls", ".XLSX", ".XLS")):
        DATASHEET_MANAGER.load_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(
            f"Unsupported file format for {file_path}. Use CSV or Excel files."
        )


## Tool definitions
### File System Tools


@tool
def get_file_list() -> str:
    """
    List available data files in the memory_files directory.
    """
    return FILESYSTEM_MANAGER.list_files()


@tool
def get_file_content(file_name: str) -> str:
    """
    Load a file from the memory_files directory.
    """
    return FILESYSTEM_MANAGER.read_file(
        file_name
    )  # TODO: switch later for detailed view


### Datasheet Tools


@tool
def get_full_dataframe_string_tool(params: Union[DatasheetLoadParams, Dict]) -> str:
    """
    Returns the *entire* loaded dataframe as a string.
    WARNING: This can produce very large output for large datasheets, potentially exceeding token limits.
    Use 'get_datasheet_chunk' or 'calculate_statistics_tool' for summaries when possible.
    Returns an error message if no data is loaded.
    """
    if isinstance(params, dict):
        params = DatasheetLoadParams(**params)

    if params.file_path and DATASHEET_MANAGER.df_filepath != params.file_path:
        _read_datasheet(params.file_path, params.sheet_name)

    try:
        return DATASHEET_MANAGER.df_as_str()
    except Exception as e:
        logger.error(
            f"[get_full_dataframe_string_tool] Error converting dataframe to string: {e}"
        )
        logger.error(traceback.format_exc())
        return f"Error converting dataframe to string: {e}"


@tool
def get_datasheet_chunk(params: Union[DatasheetChunkParams, Dict]) -> str:
    """
    Extract a specific subset of the data from the datasheet.

    Args:
        params: DataChunkParams with optional file_path, rows, and columns specifications

    Returns:
        String representation of the requested data chunk
    """
    try:
        if isinstance(params, dict):
            params = DatasheetChunkParams(**params)
        # logger.info(f"[get_datasheet_chunk] Loading data from {params.file_path} with sheet name {params.sheet_name}")
        # logger.info(f"[get_datasheet_chunk] Loading data with rows: {params.rows} and columns: {params.columns}")
        if params.file_path and DATASHEET_MANAGER.df_filepath != params.file_path:
            _read_datasheet(params.file_path, params.sheet_name)

        chunk = DATASHEET_MANAGER.get_chunk(rows=params.rows, columns=params.columns)
        return chunk.to_string()
    except Exception as e:
        logger.error(f"[get_datasheet_chunk] Error retrieving data chunk: {e}")
        logger.error(traceback.format_exc())
        return f"Error retrieving data chunk: {str(e)}"


@tool
def calculate_datasheet_statistics(
    params: Union[DatasheetStatsReqParams, Dict],
) -> Dict[str, Any]:
    """
    Calculate statistical measures for specified columns in the datasheet.

    Args:
        params: StatisticsRequest with columns to analyze and optional file_path

    Returns:
        Dictionary of calculated statistics by column
    """
    try:
        if isinstance(params, dict):
            params = DatasheetStatsReqParams(**params)

        if params.file_path and DATASHEET_MANAGER.df_filepath != params.file_path:
            _read_datasheet(params.file_path, params.sheet_name)

        return DATASHEET_MANAGER.calculate_statistics(
            columns=params.columns, rows=params.rows, stats=params.stats
        )
    except Exception as e:
        logger.error(
            f"[calculate_datasheet_statistics] Error calculating statistics: {e}"
        )
        logger.error(traceback.format_exc())
        return {"error": str(e)}


### Google API
@tool
def download_google_file(params: GoogleFileInput) -> str:
    """
    Download a file from Google Drive or export a Google Sheet as CSV.

    This tool handles both Google Drive files and Google Sheets. For Google Sheets,
    it exports the data as a CSV file. The tool automatically uses the original
    filename if no output filename is specified and ensures unique filenames by
    appending a counter if a file with the same name already exists.

    Examples:
        - Download a regular file from Google Drive
        - Export a Google Sheet as CSV with specific cell range

    Args:
        params: GoogleFileInput containing file_url, optional output_filename,
                   and optional sheet_range (for Google Sheets only)

    Returns:
        Path to the downloaded file or error message
    """
    try:
        if isinstance(params, dict):
            params = GoogleFileInput(**params)

        # Determine if the URL is for a Google Sheet
        is_sheet = (
            "spreadsheets" in params.file_url or "sheets.google.com" in params.file_url
        )

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
                file_id_or_url=params.file_url, output_file=params.output_filename
            )
            file_type = "Google Drive file"

        return f"Successfully downloaded {file_type} to: {filepath}"

    except Exception as e:
        return f"Error downloading file: {str(e)}"


# @tool
# def generate_datasheet_description(request: DataDescriptionRequest) -> str:
#     """
#     Generate a descriptive summary of the datasheet using an LLM.
#
#     Args:
#         request: DataDescriptionRequest with optional file_path and parameters
#         llm: The language model to use for description generation
#
#     Returns:
#         LLM-generated description of the data
#     """
#     try:
#         if request.file_path:
#             _ensure_file_loaded(request.file_path)
#
#         return DATASHEET_MANAGER.generate_data_description(
#             llm=llm,
#             sample_rows=request.sample_rows,
#             include_stats=request.include_stats
#         )
#     except Exception as e:
#         return f"Error generating description: {str(e)}"


# # Define the tools list for LangChain
DEFINED_TOOLS = [
    get_file_list,
    get_file_content,
    get_full_dataframe_string_tool,
    get_datasheet_chunk,
    calculate_datasheet_statistics,
    download_google_file,
]
#
DEFINED_TOOLS_DICT = {tool.name: tool for tool in DEFINED_TOOLS}
# DEFINED_TOOL_NODE = ToolNode(DEFINED_TOOLS)
