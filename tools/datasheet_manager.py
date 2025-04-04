# import pandas as pd
# import numpy as np
# from typing import List, Dict, Any, Optional, Union
# from pydantic import BaseModel, Field  # Pydantic v1
# import gspread
# from langchain.tools import BaseTool, StructuredTool
# from langchain.pydantic_v1 import BaseModel as LCBaseModel  # Use LangChain's bundled Pydantic v1
#
# # Using LangChain's bundled Pydantic version for schemas
# class GoogleSheetInput(LCBaseModel):
#     """Input for loading a Google Sheet."""
#     url: str = Field(..., description="URL of the Google Sheet")
#     sheet_name: Optional[str] = Field(None, description="Name of the worksheet to load (defaults to first sheet)")
#     range: Optional[str] = Field(None, description="Cell range to load (e.g., 'A1:D10')")
#
#
# class DataFrameTool:
#     """Base class for all dataframe operations."""
#     df: Optional[pd.DataFrame] = None
#
#     @classmethod
#     def load_data(cls, file_path: str) -> str:
#         """Load data from CSV, Excel file, or Google Sheet URL."""
#         try:
#             # Check if the path is a Google Sheets URL
#             if file_path.startswith(('https://docs.google.com/spreadsheets', 'https://drive.google.com')):
#                 return cls._load_from_google_sheet(file_path)
#             elif file_path.endswith('.csv'):
#                 cls.df = pd.read_csv(file_path)
#             elif file_path.endswith(('.xlsx', '.xls')):
#                 cls.df = pd.read_excel(file_path)
#             else:
#                 return f"Unsupported file format. Please provide CSV, Excel file, or Google Sheet URL."
#
#             return f"Successfully loaded data with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"
#         except Exception as e:
#             return f"Error loading file: {str(e)}"
#
#     @classmethod
#     def _load_from_google_sheet(cls, sheet_url: str) -> str:
#         """Load data from a publicly shared Google Sheet."""
#         try:
#             # Extract the spreadsheet key from the URL
#             if '/d/' in sheet_url:
#                 # Format: https://docs.google.com/spreadsheets/d/KEY/edit
#                 sheet_key = sheet_url.split('/d/')[1].split('/')[0]
#             else:
#                 return "Invalid Google Sheet URL. Please provide a URL in the format: https://docs.google.com/spreadsheets/d/KEY/edit"
#
#             # Access the sheet without authentication (works only for public sheets)
#             client = gspread.service_account_from_dict(None)  # No auth for public sheets
#             try:
#                 # Try to open the sheet without authentication
#                 sheet = client.open_by_key(sheet_key)
#                 worksheet = sheet.get_worksheet(0)  # Get the first worksheet
#                 data = worksheet.get_all_values()
#
#                 # Convert to DataFrame
#                 cls.df = pd.DataFrame(data[1:], columns=data[0])
#
#                 return f"Successfully loaded Google Sheet with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"
#
#             except gspread.exceptions.APIError:
#                 # If direct access fails, try with public sheet URL
#                 return cls._load_from_public_sheet(sheet_key)
#
#         except Exception as e:
#             return f"Error loading Google Sheet: {str(e)}"
#
#     @classmethod
#     def _load_from_public_sheet(cls, sheet_key: str) -> str:
#         """Load data from a publicly shared Google Sheet using the published CSV URL."""
#         try:
#             # For public sheets, you can use the CSV export URL
#             csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv"
#             cls.df = pd.read_csv(csv_url)
#             return f"Successfully loaded public Google Sheet with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"
#         except Exception as e:
#             return f"Error loading public Google Sheet: {str(e)}. Make sure the sheet is published to the web and accessible to anyone with the link."
#
#     @classmethod
#     def load_google_sheet(cls, url: str, sheet_name: Optional[str] = None, range: Optional[str] = None) -> str:
#         """Load a specific sheet or range from a Google Sheet."""
#         try:
#             # Extract the spreadsheet key from the URL
#             if '/d/' in url:
#                 sheet_key = url.split('/d/')[1].split('/')[0]
#             else:
#                 return "Invalid Google Sheet URL"
#
#             # Try to access using the CSV export approach for public sheets
#             export_url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv"
#
#             # If a specific sheet is requested, add it to the URL
#             if sheet_name:
#                 export_url += f"&gid={sheet_name}"
#
#             # Load the data
#             cls.df = pd.read_csv(export_url)
#
#             # Apply range filtering if specified
#             if range:
#                 # This is simplified; you'd need to parse A1 notation
#                 # and convert to row/column indices
#                 pass
#
#             return f"Successfully loaded Google Sheet with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"
#
#         except Exception as e:
#             return f"Error loading Google Sheet: {str(e)}"
#
#
# class DescribeDataInput(LCBaseModel):
#     """Input for describing data."""
#     columns: Optional[List[str]] = Field(None,
#                                          description="Optional list of columns to describe. If not provided, all columns will be described.")
#
#
# class FilterDataInput(LCBaseModel):
#     """Input for filtering data."""
#     query: str = Field(...,
#                        description="Query string in the format that pandas understands, e.g., 'column > 5 and other_column == \"value\"'")
#
#
# class AggregateDataInput(LCBaseModel):
#     """Input for aggregating data."""
#     column: str = Field(..., description="Column to aggregate")
#     operation: str = Field(...,
#                            description="Operation to perform: 'mean', 'median', 'sum', 'min', 'max', 'count', 'std', 'var'")
#     group_by: Optional[List[str]] = Field(None, description="Optional columns to group by")
#
#
# # Function implementations for tool usage
# def load_data_func(file_path: str) -> str:
#     """Load data from a CSV, Excel file, or Google Sheet URL."""
#     return DataFrameTool.load_data(file_path)
#
# def load_google_sheet_func(url: str, sheet_name: Optional[str] = None, range: Optional[str] = None) -> str:
#     """Load data from a Google Sheet with specific sheet name and range."""
#     return DataFrameTool.load_google_sheet(url, sheet_name, range)
#
# def describe_data_func(columns: Optional[List[str]] = None) -> str:
#     """Get statistical description of the data."""
#     if DataFrameTool.df is None:
#         return "No data loaded. Please load data first."
#
#     try:
#         if columns:
#             description = DataFrameTool.df[columns].describe().to_string()
#         else:
#             description = DataFrameTool.df.describe().to_string()
#         return f"Statistical description of the data:\n{description}"
#     except Exception as e:
#         return f"Error describing data: {str(e)}"
#
# def filter_data_func(query: str) -> str:
#     """Filter data based on a query."""
#     if DataFrameTool.df is None:
#         return "No data loaded. Please load data first."
#
#     try:
#         filtered_df = DataFrameTool.df.query(query)
#         preview = filtered_df.head(5).to_string()
#         return f"Filtered to {len(filtered_df)} rows. Preview:\n{preview}"
#     except Exception as e:
#         return f"Error filtering data: {str(e)}"
#
# def aggregate_data_func(column: str, operation: str, group_by: Optional[List[str]] = None) -> str:
#     """Aggregate data with various operations."""
#     if DataFrameTool.df is None:
#         return "No data loaded. Please load data first."
#
#     operations = {
#         'mean': np.mean,
#         'median': np.median,
#         'sum': np.sum,
#         'min': np.min,
#         'max': np.max,
#         'count': len,
#         'std': np.std,
#         'var': np.var
#     }
#
#     if operation not in operations:
#         return f"Unsupported operation. Choose from: {', '.join(operations.keys())}"
#
#     try:
#         if group_by:
#             result = DataFrameTool.df.groupby(group_by)[column].agg(operations[operation])
#             return f"Grouped {operation} of {column} by {', '.join(group_by)}:\n{result.to_string()}"
#         else:
#             result = operations[operation](DataFrameTool.df[column])
#             return f"The {operation} of {column} is: {result}"
#     except Exception as e:
#         return f"Error aggregating data: {str(e)}"
#
#
# # Create LangChain tools using StructuredTool for better Pydantic v1 compatibility
# load_data_tool = StructuredTool.from_function(
#     func=load_data_func,
#     name="load_data_tool",
#     description="Load data from a CSV, Excel file, or Google Sheet URL"
# )
#
# load_google_sheet_tool = StructuredTool.from_function(
#     func=load_google_sheet_func,
#     name="load_google_sheet_tool",
#     description="Load data from a Google Sheet with specific sheet name and range"
# )
#
# describe_data_tool = StructuredTool.from_function(
#     func=describe_data_func,
#     name="describe_data_tool",
#     description="Get statistical description of the data"
# )
#
# filter_data_tool = StructuredTool.from_function(
#     func=filter_data_func,
#     name="filter_data_tool",
#     description="Filter data based on a query"
# )
#
# aggregate_data_tool = StructuredTool.from_function(
#     func=aggregate_data_func,
#     name="aggregate_data_tool",
#     description="Aggregate data with various operations"
# )
#
# # Create a tools list that can be used with an agent
# datasheet_tools = [
#     load_data_tool,
#     load_google_sheet_tool,
#     describe_data_tool,
#     filter_data_tool,
#     aggregate_data_tool
# ]