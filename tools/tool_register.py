# from langchain.agents import Tool
# from langchain.tools import BaseTool
#
# # Import the tools from your other file
# from tools.datasheet_manager import DataFrameTool, GoogleSheetInput, DescribeDataInput, FilterDataInput, AggregateDataInput, describe_data, filter_data, aggregate_data
# from tools.file_manager import FilesystemTool
#
# # Create the filesystem tool instance
# filesystem_tool = FilesystemTool()
#
# # Define the tools list for LangChain
DEFINED_TOOLS = [
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
    # Tool(
    #     name="FileSystem",
    #     description="List, search, and load files from the memory_files directory. Pass a filename to load it, or a search term to find files. Leave empty to list all files.",
    #     func=filesystem_tool._run
    # )
]
#
DEFINED_TOOLS_DICT = {tool.name: tool for tool in DEFINED_TOOLS}