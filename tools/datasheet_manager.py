# Standard library imports
from typing import Union, Optional, List, Dict, Any, Tuple, Iterable

# Third-party library imports
from langchain_core.language_models.chat_models import BaseChatModel
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


### Main class
class DatasheetManager:
    """
    A class to manage and interact with datasheet files (CSV, Excel) with various operations
    including data extraction, description, and statistical analysis.
    """

    _df: Optional[pd.DataFrame] = None
    _df_filepath = None

    def get_sheet_names(self, file_path: str) -> List[str]:
        """
        Get list of sheet names from an Excel file without loading the data.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of sheet names in the Excel file
        """
        try:
            # Using pd.ExcelFile for efficient sheet name extraction
            with pd.ExcelFile(file_path) as excel_file:
                return excel_file.sheet_names
        except Exception as e:
            print(f"Error reading sheet names from {file_path}: {e}")
            return []


    def load_csv(self, file_path: str, **kwargs) -> None:
        """
        Load data from a CSV file.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv
        """
        self._df = pd.read_csv(file_path, **kwargs)
        self._df_filepath = file_path
        print(
            f"Loaded CSV file from {file_path} with {len(self._df)} rows and {len(self._df.columns)} columns"
        )

    def load_excel(self, file_path: str, sheet_name=0, show_sheets: bool = False, **kwargs) -> None:
        """
        Load data from an Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name or index of sheet to load (default: 0, first sheet)
            show_sheets: Whether to display available sheet names (default: False)
            **kwargs: Additional arguments to pass to pandas.read_excel
        """
        # Get available sheet names
        available_sheets = self.get_sheet_names(file_path)

        if show_sheets and available_sheets:
            print(f"Available sheets in {file_path}: {available_sheets}")

        # Validate sheet_name if it's a string
        if isinstance(sheet_name, str) and sheet_name not in available_sheets:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}")

        # Validate sheet index if it's an integer
        if isinstance(sheet_name, int) and (sheet_name >= len(available_sheets) or sheet_name < 0):
            raise IndexError(f"Sheet index {sheet_name} out of range. Available sheets: {len(available_sheets)}")

        self._df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        self._df_filepath = file_path

        # Get actual sheet name for display
        actual_sheet = available_sheets[sheet_name] if isinstance(sheet_name, int) else sheet_name
        print(
            f"Loaded Excel file from {file_path} (sheet: {actual_sheet}) with {len(self._df)} rows and {len(self._df.columns)} columns"
        )

    def load_google_sheet(self, sheet_url: str) -> None:
        """
        Load data from a Google Sheet.

        Args:
            sheet_url: URL of the Google Sheet

        Note: Implementation to be added later
        """
        # Implementation to be added later
        self._df_filepath = sheet_url
        raise NotImplementedError("Google Sheet loading is not implemented yet.")

    ## Note: probably not needed
    # def save_to_csv(self, file_path: str, **kwargs) -> None:
    #     """
    #     Save the dataframe to a CSV file.
    #
    #     Args:
    #         file_path: Path to save the CSV file
    #         **kwargs: Additional arguments to pass to pandas.to_csv
    #     """
    #     if self.df is not None:
    #         self.df.to_csv(file_path, **kwargs)
    #         print(f"Saved dataframe to CSV file: {file_path}")
    #     else:
    #         print("No data to save. Please load data first.")
    #
    # def save_to_excel(self, file_path: str, **kwargs) -> None:
    #     """
    #     Save the dataframe to an Excel file.
    #
    #     Args:
    #         file_path: Path to save the Excel file
    #         **kwargs: Additional arguments to pass to pandas.to_excel
    #     """
    #     if self.df is not None:
    #         self.df.to_excel(file_path, **kwargs)
    #         print(f"Saved dataframe to Excel file: {file_path}")
    #     else:
    #         print("No data to save. Please load data first.")

    def get_description(self) -> Dict[str, Any]:
        """
        Return a description of the dataframe including columns and sizes.

        Returns:
            Dictionary containing dataframe metadata
        """
        if self._df is None:
            return {"error": "No data loaded. Please load data first."}

        description = {
            "rows": len(self._df),
            "columns": len(self._df.columns),
            "column_names": list(self._df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self._df.dtypes.items()},
            "memory_usage": self._df.memory_usage(deep=True).sum(),
            "missing_values": self._df.isna().sum().to_dict(),
        }

        return description

    def get_chunk(
        self,
        rows: Optional[Union[List[int], List[str], slice, int, str]] = None,
        columns: Optional[Union[List[int], List[str], slice, int, str]] = None,
    ) -> pd.DataFrame:
        """
        Extract a specific chunk of data from the dataframe.

        Args:
            rows: Row indices, names, or slice to select
            columns: Column indices, names, or slice to select

        Returns:
            Pandas DataFrame containing the requested data chunk
        """
        if self._df is None:
            print("No data loaded. Please load data first.")
            return pd.DataFrame()

        # Default to all rows and columns if not specified
        if rows is None and columns is None:
            return self._df.copy()

        # Handle rows
        if rows is not None:
            if isinstance(rows, (int, str)):
                rows = [rows]
            row_data = self._df.loc[rows]
        else:
            row_data = self._df

        # Handle columns
        if columns is not None:
            if isinstance(columns, (int, str)):
                columns = [columns]
            return row_data[columns]

        return row_data

    def generate_data_description(
        self, llm: BaseChatModel, sample_rows: int = 5, include_stats: bool = True
    ) -> str:
        """
        Generate a descriptive summary of the data using an LLM.

        Args:
            llm: LangChain ChatModel instance
            sample_rows: Number of sample rows to include (default: 5)
            include_stats: Whether to include basic statistics (default: True)

        Returns:
            String containing the LLM-generated description
        """
        if self._df is None:
            return "No data loaded. Please load data first."

        # Prepare sample data and statistics
        sample_data = self._df.head(sample_rows).to_string()

        # Basic statistics for numeric columns
        stats_text = ""
        if include_stats:
            numeric_cols = self._df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                stats = self._df[numeric_cols].describe().to_string()
                stats_text = f"\n\nStatistics for numeric columns:\n{stats}"

        # Prepare the prompt for the LLM
        # TODO: Move the prompt to tool_prompts_texts.py
        prompt = f"""
        Please analyze the following dataset and provide a brief description of what it contains.

        Dataset has {len(self._df)} rows and {len(self._df.columns)} columns.
        Columns: {', '.join(self._df.columns.tolist())}

        Here's a sample of the data:
        {sample_data}
        {stats_text}

        Please describe:
        1. What kind of data this appears to be
        2. The main entities or concepts represented
        3. What insights might be extracted from this data
        """

        # Send to LLM and get response
        response = llm.invoke(prompt)
        return response.content

    def calculate_statistics(
        self,
        columns: Union[str, List[str]],
        rows: Optional[Union[List[int], List[str], slice]] = None,
        stats: Iterable[str] = ("mean", "median", "std", "min", "max"),
    ) -> Dict[str, Any]:
        """
        Perform statistical calculations on specified chunks of data using NumPy for performance.

        Args:
            columns: Column(s) to calculate statistics for
            rows: Optional row subset to use (default: all rows)
            stats: List of statistics to calculate (default: mean, median, std, min, max)

        Returns:
            Dictionary of calculated statistics
        """
        if self._df is None:
            return {"error": "No data loaded. Please load data first."}

        # Get the data subset
        if rows is not None:
            data = self._df.loc[rows]
        else:
            data = self._df

        if isinstance(columns, str):
            columns = [columns]

        # Filter to only numeric columns
        numeric_cols = [
            col for col in columns if pd.api.types.is_numeric_dtype(data[col])
        ]
        if not numeric_cols:
            return {"error": "No numeric columns found in the specified columns."}

        results = {}
        for col in numeric_cols:
            # Convert to NumPy array for faster calculations
            # Drop NaN values for accurate calculations
            arr = data[col].dropna().to_numpy()

            if len(arr) == 0:
                results[col] = {"error": "No non-NA values in column"}
                continue

            col_stats = {}

            for stat in stats:
                # TODO: Rewrite this to use a dictionary of functions
                if stat == "mean":
                    col_stats["mean"] = np.mean(arr)
                elif stat == "median":
                    col_stats["median"] = np.median(arr)
                elif stat == "std":
                    col_stats["std"] = np.std(
                        arr, ddof=1
                    )  # Using ddof=1 to match pandas default
                elif stat == "min":
                    col_stats["min"] = np.min(arr)
                elif stat == "max":
                    col_stats["max"] = np.max(arr)
                elif stat == "count":
                    col_stats["count"] = len(arr)
                elif stat == "sum":
                    col_stats["sum"] = np.sum(arr)
                elif stat == "variance" or stat == "var":
                    col_stats["variance"] = np.var(
                        arr, ddof=1
                    )  # Using ddof=1 to match pandas default
                elif stat == "range":
                    col_stats["range"] = np.max(arr) - np.min(arr)

            results[col] = col_stats

        return results

    def df_as_str(self, limit_length: Optional[int] = None) -> str:
        """
        Return the dataframe as a string representation.
        """
        if self._df is None:
            return "No data loaded. Please load data first."
        return (
            self._df.to_string()
            if limit_length is None
            else self._df.to_string()[:limit_length]
        )

    @property
    def df_filepath(self):
        """
        Return the filename of the loaded dataframe.
        """
        return self._df_filepath


### Pydantic models
class DatasheetLoadParams(BaseModel):
    file_path: str = Field(..., description="Path to the file to load")
    sheet_name: Optional[Union[str, int]] = Field(
        0, description="Sheet name or index for Excel files"
    )


class DatasheetChunkParams(BaseModel):
    file_path: Optional[str] = Field(
        None, description="Path to the file to load (if not already loaded)"
    )
    sheet_name: Optional[Union[str, int]] = Field(
        0, description="Sheet name or index for Excel files (default: 0, first sheet)"
    )
    rows: Optional[Union[List[int], List[str], int, str]] = Field(
        None, description="Row indices, names, or slice to select"
    )
    columns: Optional[Union[List[int], List[str], int, str]] = Field(
        None, description="Column indices, names, or slice to select"
    )


class DatasheetStatsReqParams(BaseModel):
    file_path: Optional[str] = Field(
        None, description="Path to the file to load (if not already loaded)"
    )
    columns: Union[str, List[str]] = Field(
        ..., description="Column(s) to calculate statistics for"
    )
    rows: Optional[Union[List[int], List[str]]] = Field(
        None, description="Optional row subset to use (default: all rows)"
    )
    stats: Optional[List[str]] = Field(
        ["mean"],
        description="Statistics to calculate (e.g., mean, median, std, min, max, count, sum, variance, range)",
    )


### Singleton instance
DATASHEET_MANAGER = DatasheetManager()
# TODO: Think if this should be a singleton or not (Think if it should handle multiple datasheets at once [Memory])
