# import os
# from typing import List, Dict, Optional
# from pydantic import BaseModel, Field
# import json
# from langchain.agents import Tool
# from langchain.tools import BaseTool
# from langchain.callbacks.manager import CallbackManagerForToolRun
#
# from tools.datasheet_manager import DataFrameTool
#
#
# class MemoryFileInfo(BaseModel):
#     """Information about a file in memory."""
#     filename: str = Field(..., description="Name of the file")
#     path: str = Field(..., description="Full path to the file")
#     description: Optional[str] = Field(None, description="Short description of the file content")
#
#
# class FileSearchInput(BaseModel):
#     """Input for searching files."""
#     query: str = Field(..., description="Search query to find files by name")
#
#
# class FilesystemTool(BaseTool):
#     """Tool for interacting with files in the memory_files directory."""
#     name: str = "filesystem_tool"
#     description: str = "List files in memory_files directory and search for specific files"
#     memory_dir: str = "memory_files"
#     memory_index_file: str = "memory_files/index.json"
#
#     def _load_memory_index(self) -> Dict[str, MemoryFileInfo]:
#         """Load the memory index from disk."""
#         if os.path.exists(self.memory_index_file):
#             try:
#                 with open(self.memory_index_file, 'r') as f:
#                     data = json.load(f)
#                     return {k: MemoryFileInfo(**v) for k, v in data.items()}
#             except Exception as e:
#                 print(f"Error loading memory index: {str(e)}")
#                 return {}
#         else:
#             return {}
#
#     def _save_memory_index(self, index: Dict[str, MemoryFileInfo]) -> None:
#         """Save the memory index to disk."""
#         # Ensure directory exists
#         os.makedirs(os.path.dirname(self.memory_index_file), exist_ok=True)
#
#         try:
#             with open(self.memory_index_file, 'w') as f:
#                 # Convert MemoryFileInfo objects to dictionaries
#                 serializable_index = {k: v.dict() for k, v in index.items()}
#                 json.dump(serializable_index, f, indent=2)
#         except Exception as e:
#             print(f"Error saving memory index: {str(e)}")
#
#     def _update_file_description(self, file_path: str, description: str = None) -> None:
#         """Update or add a file description in the memory index."""
#         memory_index = self._load_memory_index()
#         filename = os.path.basename(file_path)
#
#         if description is None and DataFrameTool.df is not None:
#             # Generate a description if not provided and a dataframe is loaded
#             try:
#                 num_rows = len(DataFrameTool.df)
#                 num_cols = len(DataFrameTool.df.columns)
#                 columns = ", ".join(DataFrameTool.df.columns.tolist()[:5])
#                 if len(DataFrameTool.df.columns) > 5:
#                     columns += f" and {len(DataFrameTool.df.columns) - 5} more"
#
#                 description = f"Dataset with {num_rows} rows and {num_cols} columns. Columns include: {columns}."
#             except:
#                 description = "File loaded but unable to generate automatic description."
#
#         # Update or add the file info
#         memory_index[filename] = MemoryFileInfo(
#             filename=filename,
#             path=file_path,
#             description=description
#         )
#
#         # Save the updated index
#         self._save_memory_index(memory_index)
#
#     def _run(self, query: str = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
#         """Run the filesystem tool."""
#         # Ensure the memory directory exists
#         os.makedirs(self.memory_dir, exist_ok=True)
#
#         # If no query, list all files
#         if not query:
#             files = os.listdir(self.memory_dir)
#             file_list = [f for f in files if os.path.isfile(os.path.join(self.memory_dir, f)) and f != "index.json"]
#
#             if not file_list:
#                 return "No files found in memory_files directory."
#
#             # Get descriptions from memory index
#             memory_index = self._load_memory_index()
#             result = "Files in memory:\n"
#
#             for file in file_list:
#                 description = memory_index.get(file, MemoryFileInfo(filename=file, path=os.path.join(self.memory_dir,
#                                                                                                      file))).description
#                 description_text = f" - {description}" if description else ""
#                 result += f"- {file}{description_text}\n"
#
#             return result
#
#         # If query looks like a filename, try to load it
#         if "." in query and "/" not in query and "\\" not in query:
#             file_path = os.path.join(self.memory_dir, query)
#             if os.path.exists(file_path):
#                 # Load the file using DataFrameTool
#                 result = DataFrameTool.load_data(file_path)
#
#                 # Update the file description in memory
#                 self._update_file_description(file_path)
#
#                 return f"Found and loaded file: {query}\n{result}"
#             else:
#                 return f"File '{query}' not found in memory_files directory."
#
#         # Otherwise, search for files matching the query
#         files = os.listdir(self.memory_dir)
#         matching_files = [f for f in files if
#                           query.lower() in f.lower() and os.path.isfile(os.path.join(self.memory_dir, f))]
#
#         if not matching_files:
#             return f"No files matching '{query}' found in memory_files directory."
#
#         # Get descriptions from memory index
#         memory_index = self._load_memory_index()
#         result = f"Files matching '{query}':\n"
#
#         for file in matching_files:
#             description = memory_index.get(file, MemoryFileInfo(filename=file,
#                                                                 path=os.path.join(self.memory_dir, file))).description
#             description_text = f" - {description}" if description else ""
#             result += f"- {file}{description_text}\n"
#
#         return result
#
#
# # Create a LangChain tool
# filesystem_tool = Tool(
#     name="FileSystem",
#     description="List, search, and load files from the memory_files directory. Pass a filename to load it, or a search term to find files.",
#     func=FilesystemTool()._run
# )