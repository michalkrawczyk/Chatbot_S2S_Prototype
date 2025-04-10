from langchain.tools import Tool
from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from langgraph.prebuilt import ToolNode
#
# # Import the tools from your other file
# from tools.datasheet_manager import DataFrameTool, GoogleSheetInput, DescribeDataInput, FilterDataInput, AggregateDataInput, describe_data, filter_data, aggregate_data
from tools.file_manager import FileSystemManager
#
# Create the filesystem tool instance
filesystem_manager = FileSystemManager()

@tool
def get_file_list(file_name: str = "") -> str:
    """
    List available data files in the memory_files directory.
    """
    return filesystem_manager.list_files

@tool
def get_file_content(file_name: str) -> str:
    """
    Load a file from the memory_files directory.
    """
    return filesystem_manager.read_file(file_name) #TODO: switch later for detailed view

#
# # Define the tools list for LangChain
DEFINED_TOOLS = [
    get_file_list,
    get_file_content,
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
