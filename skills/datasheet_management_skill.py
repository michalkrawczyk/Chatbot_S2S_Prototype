"""
---
name: datasheet_management
version: 1.0.0
description: |
  Provides operations for working with datasheets including CSV and Excel files.
  Enables reading, querying, and analyzing structured data with statistical
  operations and chunked access to large datasets.
author: Chatbot Agent System
tags:
  - datasheet
  - csv
  - excel
  - data
  - statistics
  - analytics
created_at: 2026-01-19
reference: https://code.claude.com/docs/en/skills
---

Datasheet Management Skill

This skill provides comprehensive datasheet operations for the agent, including:
- Loading CSV and Excel files
- Retrieving full datasheet content
- Chunked access for large datasets
- Statistical analysis (mean, median, sum, count, etc.)

The skill abstracts the complexity of working with pandas DataFrames and provides
a clean interface for the agent to query and analyze structured data.

Usage Example:
    from skills.datasheet_management_skill import DatasheetManagementSkill
    
    skill = DatasheetManagementSkill()
    tools = skill.tools  # Get the datasheet tools
    
    # Use with agent
    agent.bind_tools(tools)

Best Practices:
- Use chunked access for large files to avoid token limits
- Use statistics tools for summaries instead of full dataframe
- Load files once and reuse the loaded data for multiple queries

Reference: Claude Skills Methodology - Data Management Patterns
https://code.claude.com/docs/en/skills#data-management
"""

from typing import List, Any, Union, Dict
from langchain_core.tools import tool

from skills.base_skill import BaseSkill, SkillMetadata
from tools.datasheet_manager import (
    DATASHEET_MANAGER,
    DatasheetLoadParams,
    DatasheetChunkParams,
    DatasheetStatsReqParams,
)
import logging
import traceback


logger = logging.getLogger("DatasheetManagementSkill")


class DatasheetManagementSkill(BaseSkill):
    """
    Skill for datasheet operations including CSV and Excel file handling.
    
    This skill provides comprehensive tools for working with structured data
    in CSV and Excel formats. It supports loading files, retrieving data in
    chunks to manage token limits, and calculating statistical measures.
    
    Capabilities:
    - Load CSV and Excel files
    - Retrieve full dataframe as string (with token limit warnings)
    - Get specific chunks of data (rows/columns)
    - Calculate statistics on columns (mean, median, sum, count, etc.)
    
    Performance Considerations:
    - Full dataframe retrieval can exceed token limits for large files
    - Use chunked access for files with many rows
    - Statistics provide efficient summaries without loading all data
    
    Error Handling:
    - All operations wrapped in try-except blocks
    - Descriptive error messages returned instead of exceptions
    - Errors logged for debugging
    
    Reference: Claude Skills Methodology - Performance & Error Handling
    https://code.claude.com/docs/en/skills#performance
    """
    
    def __init__(self):
        """Initialize the datasheet management skill."""
        super().__init__()
    
    def _define_metadata(self) -> SkillMetadata:
        """Define metadata for the datasheet management skill."""
        return SkillMetadata(
            name="datasheet_management",
            version="1.0.0",
            description=(
                "Provides operations for working with datasheets including CSV and "
                "Excel files, with support for querying, chunking, and statistics"
            ),
            author="Chatbot Agent System",
            tags=["datasheet", "csv", "excel", "data", "statistics", "analytics"],
        )
    
    def _register_tools(self) -> List[Any]:
        """Register datasheet management tools."""
        
        def _read_datasheet(file_path: str, sheet_name: Union[str, int, None] = None):
            """Helper to load a datasheet file."""
            if file_path.endswith((".csv", ".CSV")):
                DATASHEET_MANAGER.load_csv(file_path)
            elif file_path.endswith((".xlsx", ".xls", ".XLSX", ".XLS")):
                DATASHEET_MANAGER.load_excel(file_path, sheet_name=sheet_name)
            else:
                raise ValueError(
                    f"Unsupported file format for {file_path}. Use CSV or Excel files."
                )
        
        @tool
        def get_full_dataframe_string_tool(params: Union[DatasheetLoadParams, Dict]) -> str:
            """
            Returns the *entire* loaded dataframe as a string.
            
            WARNING: This can produce very large output for large datasheets,
            potentially exceeding token limits. Use 'get_datasheet_chunk' or
            'calculate_statistics_tool' for summaries when possible.
            
            Args:
                params: DatasheetLoadParams with file_path and optional sheet_name
                
            Returns:
                String representation of the entire dataframe, or error message
                
            Example:
                >>> params = {"file_path": "data.csv"}
                >>> get_full_dataframe_string_tool(params)
                '   Name  Age  Score\\n0  Alice   25     95\\n...'
                
            Best Practice:
                Prefer chunked access or statistics for large files to avoid
                token limit issues.
                
            Reference: Claude Skills Methodology - Large Data Handling
            https://code.claude.com/docs/en/skills#large-data
            """
            if isinstance(params, dict):
                params = DatasheetLoadParams(**params)
            
            if params.file_path and DATASHEET_MANAGER.df_filepath != params.file_path:
                try:
                    _read_datasheet(params.file_path, params.sheet_name)
                except Exception as e:
                    return f"Error loading datasheet: {str(e)}"
            
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
            
            This is the recommended way to access large datasheets, as it allows
            retrieving only the necessary rows and columns, avoiding token limits.
            
            Args:
                params: DatasheetChunkParams with:
                    - file_path: Path to the datasheet file
                    - rows: Optional list of row indices or slice (e.g., [0, 10])
                    - columns: Optional list of column names
                    - sheet_name: Optional sheet name for Excel files
            
            Returns:
                String representation of the requested data chunk, or error message
                
            Example:
                >>> params = {
                ...     "file_path": "data.csv",
                ...     "rows": [0, 5],  # First 5 rows
                ...     "columns": ["Name", "Score"]  # Only these columns
                ... }
                >>> get_datasheet_chunk(params)
                '   Name  Score\\n0  Alice     95\\n...'
                
            Performance:
                Chunked access is much more efficient for large files and helps
                stay within token limits for LLM context.
                
            Reference: Claude Skills Methodology - Chunked Data Access
            https://code.claude.com/docs/en/skills#chunking
            """
            try:
                if isinstance(params, dict):
                    params = DatasheetChunkParams(**params)
                
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
            
            This provides efficient summaries of data without loading the entire
            dataframe into the context. Supports common statistical operations
            like mean, median, sum, count, min, max, and std deviation.
            
            Args:
                params: DatasheetStatsReqParams with:
                    - file_path: Path to the datasheet file
                    - columns: List of column names to analyze
                    - rows: Optional row range
                    - stats: Optional list of specific statistics to calculate
                    - sheet_name: Optional sheet name for Excel files
            
            Returns:
                Dictionary of calculated statistics by column, or error dict
                
            Example:
                >>> params = {
                ...     "file_path": "sales.csv",
                ...     "columns": ["Revenue", "Profit"],
                ...     "stats": ["mean", "sum", "count"]
                ... }
                >>> calculate_datasheet_statistics(params)
                {
                    'Revenue': {'mean': 50000, 'sum': 500000, 'count': 10},
                    'Profit': {'mean': 15000, 'sum': 150000, 'count': 10}
                }
                
            Supported Statistics:
                - mean: Average value
                - median: Middle value
                - sum: Total sum
                - count: Number of values
                - min: Minimum value
                - max: Maximum value
                - std: Standard deviation
                
            Performance:
                Statistics are calculated efficiently without loading full data
                into context, making this ideal for large datasets.
                
            Reference: Claude Skills Methodology - Statistical Summaries
            https://code.claude.com/docs/en/skills#statistics
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
        
        # Return list of tools
        return [
            get_full_dataframe_string_tool,
            get_datasheet_chunk,
            calculate_datasheet_statistics,
        ]
