from typing import Union, Optional, List, Dict, Any
from langchain.tools import Tool
from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from langgraph.prebuilt import ToolNode

#
# # Import the tools from your other file
# from tools.datasheet_manager import DataFrameTool, GoogleSheetInput, DescribeDataInput, FilterDataInput, AggregateDataInput, describe_data, filter_data, aggregate_data
from tools.file_manager import FileSystemManager
from tools.datasheet_manager import DATASHEET_MANAGER, DatasheetLoadParams, DatasheetChunkParams, DatasheetStatsReqParams

import traceback
import logging
logger = logging.getLogger('Tool Register')


#
# Create the filesystem tool instance
FILESYSTEM_MANAGER = FileSystemManager()

def _read_datasheet(file_path: str, sheet_name: Optional[Union[str, int]] = None):
    """
    Load a datasheet file and return its content as a string.
    """
    if file_path.endswith(('.csv', '.CSV')):
        DATASHEET_MANAGER.load_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls', '.XLSX', '.XLS')):
        DATASHEET_MANAGER.load_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use CSV or Excel files.")



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
    return FILESYSTEM_MANAGER.read_file(file_name) #TODO: switch later for detailed view

### Datasheet Tools

@tool
def get_full_dataframe_string_tool(params: DatasheetLoadParams) -> str:
    """
    Returns the *entire* loaded dataframe as a string.
    WARNING: This can produce very large output for large datasheets, potentially exceeding token limits.
    Use 'get_datasheet_chunk' or 'calculate_statistics_tool' for summaries when possible.
    Returns an error message if no data is loaded.
    """
    if params.file_path and DATASHEET_MANAGER.df_filepath != params.file_path:
        _read_datasheet(params.file_path, params.sheet_name)

    try:
        return DATASHEET_MANAGER.df_as_str()
    except Exception as e:
        logger.error(f"[get_full_dataframe_string_tool] Error converting dataframe to string: {e}")
        logger.error(traceback.format_exc())
        return f"Error converting dataframe to string: {e}"

@tool
def get_datasheet_chunk(params: DatasheetChunkParams) -> str:
    """
    Extract a specific subset of the data from the datasheet.

    Args:
        params: DataChunkParams with optional file_path, rows, and columns specifications

    Returns:
        String representation of the requested data chunk
    """
    try:
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
def calculate_datasheet_statistics(params: DatasheetStatsReqParams) -> Dict[str, Any]:
    """
    Calculate statistical measures for specified columns in the datasheet.

    Args:
        params: StatisticsRequest with columns to analyze and optional file_path

    Returns:
        Dictionary of calculated statistics by column
    """
    try:
        if params.file_path and DATASHEET_MANAGER.df_filepath != params.file_path:
            _read_datasheet(params.file_path, params.sheet_name)

        return DATASHEET_MANAGER.calculate_statistics(
            columns=params.columns,
            rows=params.rows,
            stats=params.stats
        )
    except Exception as e:
        logger.error(f"[calculate_datasheet_statistics] Error calculating statistics: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

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
    # Tool(
    #     name="LoadData",
    #     description="Load data from a CSV, Excel file, or Google Sheet URL. Provide the file path or URL as input.",
    #     func=lambda file_path: DataFrameTool.load_data(file_path)
    # ),
    #
    # Tool(
    #     name="LoadGoogleSheet",
    #     description="Load data from a Google Sheet with options for sheet name and range. Input should be JSON with fields 'url', optional 'sheet_name', and optional 'range'.",
    #     func=lambda input_data: DataFrameTool.load_google_sheet(GoogleSheetInput(**input_data))
    # ),
    #
    # Tool(
    #     name="DescribeData",
    #     description="Get statistical description of loaded data. Optionally provide specific columns to describe as a JSON with 'columns' field containing a list of column names.",
    #     func=lambda input_data: describe_data(DescribeDataInput(**input_data) if input_data else DescribeDataInput())
    # ),
    #
    # Tool(
    #     name="FilterData",
    #     description="Filter data based on a query. Input should be JSON with 'query' field containing a pandas query string, e.g. 'column > 5 and other_column == \"value\"'.",
    #     func=lambda input_data: filter_data(FilterDataInput(**input_data))
    # ),
    #
    # Tool(
    #     name="AggregateData",
    #     description="Aggregate data with operations like mean, median, sum, etc. Input should be JSON with 'column', 'operation', and optional 'group_by' fields.",
    #     func=lambda input_data: aggregate_data(AggregateDataInput(**input_data))
    # ),
    #
]
#
DEFINED_TOOLS_DICT = {tool.name: tool for tool in DEFINED_TOOLS}
# DEFINED_TOOL_NODE = ToolNode(DEFINED_TOOLS)
