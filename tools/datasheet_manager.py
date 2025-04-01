import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import gspread #

class GoogleSheetInput(BaseModel):
    """Input for loading a Google Sheet."""
    url: str = Field(..., description="URL of the Google Sheet")
    sheet_name: Optional[str] = Field(None, description="Name of the worksheet to load (defaults to first sheet)")
    range: Optional[str] = Field(None, description="Cell range to load (e.g., 'A1:D10')")


class DataFrameTool:
    """Base class for all dataframe operations."""
    df: Optional[pd.DataFrame] = None

    @classmethod
    def load_data(cls, file_path: str) -> str:
        """Load data from CSV, Excel file, or Google Sheet URL."""
        try:
            # Check if the path is a Google Sheets URL
            if file_path.startswith(('https://docs.google.com/spreadsheets', 'https://drive.google.com')):
                return cls._load_from_google_sheet(file_path)
            elif file_path.endswith('.csv'):
                cls.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                cls.df = pd.read_excel(file_path)
            else:
                return f"Unsupported file format. Please provide CSV, Excel file, or Google Sheet URL."

            return f"Successfully loaded data with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"
        except Exception as e:
            return f"Error loading file: {str(e)}"

    @classmethod
    def _load_from_google_sheet(cls, sheet_url: str) -> str:
        """Load data from a publicly shared Google Sheet."""
        try:
            # Extract the spreadsheet key from the URL
            if '/d/' in sheet_url:
                # Format: https://docs.google.com/spreadsheets/d/KEY/edit
                sheet_key = sheet_url.split('/d/')[1].split('/')[0]
            else:
                return "Invalid Google Sheet URL. Please provide a URL in the format: https://docs.google.com/spreadsheets/d/KEY/edit"

            # Access the sheet without authentication (works only for public sheets)
            client = gspread.service_account_from_dict(None)  # No auth for public sheets
            try:
                # Try to open the sheet without authentication
                sheet = client.open_by_key(sheet_key)
                worksheet = sheet.get_worksheet(0)  # Get the first worksheet
                data = worksheet.get_all_values()

                # Convert to DataFrame
                cls.df = pd.DataFrame(data[1:], columns=data[0])

                return f"Successfully loaded Google Sheet with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"

            except gspread.exceptions.APIError:
                # If direct access fails, try with public sheet URL
                return cls._load_from_public_sheet(sheet_key)

        except Exception as e:
            return f"Error loading Google Sheet: {str(e)}"

    @classmethod
    def _load_from_public_sheet(cls, sheet_key: str) -> str:
        """Load data from a publicly shared Google Sheet using the published CSV URL."""
        try:
            # For public sheets, you can use the CSV export URL
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv"
            cls.df = pd.read_csv(csv_url)
            return f"Successfully loaded public Google Sheet with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"
        except Exception as e:
            return f"Error loading public Google Sheet: {str(e)}. Make sure the sheet is published to the web and accessible to anyone with the link."

    @classmethod
    def load_google_sheet(cls, input_data: GoogleSheetInput) -> str:
        """Load a specific sheet or range from a Google Sheet."""
        try:
            # Extract the spreadsheet key from the URL
            if '/d/' in input_data.url:
                sheet_key = input_data.url.split('/d/')[1].split('/')[0]
            else:
                return "Invalid Google Sheet URL"

            # Try to access using the CSV export approach for public sheets
            url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv"

            # If a specific sheet is requested, add it to the URL
            if input_data.sheet_name:
                url += f"&gid={input_data.sheet_name}"

            # Load the data
            cls.df = pd.read_csv(url)

            # Apply range filtering if specified
            if input_data.range:
                # This is simplified; you'd need to parse A1 notation
                # and convert to row/column indices
                pass

            return f"Successfully loaded Google Sheet with {len(cls.df)} rows and {len(cls.df.columns)} columns. Columns: {', '.join(cls.df.columns.tolist())}"

        except Exception as e:
            return f"Error loading Google Sheet: {str(e)}"

class DescribeDataInput(BaseModel):
    """Input for describing data."""
    columns: Optional[List[str]] = Field(None,
                                         description="Optional list of columns to describe. If not provided, all columns will be described.")


class FilterDataInput(BaseModel):
    """Input for filtering data."""
    query: str = Field(...,
                       description="Query string in the format that pandas understands, e.g., 'column > 5 and other_column == \"value\"'")


class AggregateDataInput(BaseModel):
    """Input for aggregating data."""
    column: str = Field(..., description="Column to aggregate")
    operation: str = Field(...,
                           description="Operation to perform: 'mean', 'median', 'sum', 'min', 'max', 'count', 'std', 'var'")
    group_by: Optional[List[str]] = Field(None, description="Optional columns to group by")




# Define the tools
def describe_data(input_df: DescribeDataInput) -> str:
    """Get statistical description of the data."""
    if DataFrameTool.df is None:
        return "No data loaded. Please load data first."

    try:
        if input_df.columns:
            description = DataFrameTool.df[input_df.columns].describe().to_string()
        else:
            description = DataFrameTool.df.describe().to_string()
        return f"Statistical description of the data:\n{description}"
    except Exception as e:
        return f"Error describing data: {str(e)}"


def filter_data(input_df: FilterDataInput) -> str:
    """Filter data based on a query."""
    if DataFrameTool.df is None:
        return "No data loaded. Please load data first."

    try:
        filtered_df = DataFrameTool.df.query(input_df.query)
        preview = filtered_df.head(5).to_string()
        return f"Filtered to {len(filtered_df)} rows. Preview:\n{preview}"
    except Exception as e:
        return f"Error filtering data: {str(e)}"


def aggregate_data(input_df: AggregateDataInput) -> str:
    """Aggregate data with various operations."""
    if DataFrameTool.df is None:
        return "No data loaded. Please load data first."

    operations = {
        'mean': np.mean,
        'median': np.median,
        'sum': np.sum,
        'min': np.min,
        'max': np.max,
        'count': len,
        'std': np.std,
        'var': np.var
    }

    if input_df.operation not in operations:
        return f"Unsupported operation. Choose from: {', '.join(operations.keys())}"

    try:
        if input_df.group_by:
            result = DataFrameTool.df.groupby(input_df.group_by)[input_df.column].agg(operations[input_df.operation])
            return f"Grouped {input_df.operation} of {input_df.column} by {', '.join(input_df.group_by)}:\n{result.to_string()}"
        else:
            result = operations[input_df.operation](DataFrameTool.df[input_df.column])
            return f"The {input_df.operation} of {input_df.column} is: {result}"
    except Exception as e:
        return f"Error aggregating data: {str(e)}"
