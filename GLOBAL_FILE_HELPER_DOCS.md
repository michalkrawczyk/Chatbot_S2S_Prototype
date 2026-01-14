# Global File Helper Catalog - Feature Documentation

## Overview

The Global File Helper Catalog is a centralized system that tracks all files accessed by the chatbot application, maintaining metadata including file paths, names, and LLM-generated descriptions. This feature enhances the agent's contextual awareness by providing it with a comprehensive overview of all available files.

## Architecture

### Components

1. **FileSystemManager** (`tools/file_manager.py`)
   - Manages the global file helper catalog
   - Generates LLM-based file descriptions
   - Provides context strings for agent integration

2. **AgentLLM** (`agents.py`)
   - Integrates global helper into agent context
   - Enhances file context with helper descriptions

3. **App Integration** (`app.py`)
   - Automatically adds uploaded files to global helper
   - Triggers LLM description generation

### Data Structure

The global helper is stored in `data_files/global_file_helper.json` as a JSON array:

```json
[
  {
    "file": "config.json",
    "path": "/absolute/path/to/config.json",
    "description": "Configuration file with system settings and runtime parameters",
    "origin": "user_upload",
    "last_updated": "2026-01-12T21:23:00.502578"
  },
  {
    "file": "model.py",
    "path": "/absolute/path/to/model.py",
    "description": "Defines chatbot model architecture and key operations",
    "origin": "file_scan",
    "last_updated": "2026-01-12T21:23:00.502739"
  }
]
```

### Fields

- **file**: Base filename (e.g., "config.json")
- **path**: Full absolute path to the file
- **description**: Human-readable description of the file's purpose and content
- **origin**: Source of the entry (e.g., "user_upload", "file_scan", "google_drive")
- **last_updated**: ISO 8601 timestamp of last update

## Usage

### Adding Files to Global Helper

#### Automatic (via File Upload)

When users upload files through the UI, they are automatically added to the global helper:

```python
# In app.py save_to_memory_files()
llm = AGENT.get_llm if AGENT.get_llm else None
FILESYSTEM_MANAGER.add_file_to_global_helper(
    file_path=filepath,
    origin="user_upload",
    llm_summarizer=llm,  # Generates description automatically
)
```

#### Manual (Programmatic)

```python
from tools.tool_register import FILESYSTEM_MANAGER

# Add with explicit description
FILESYSTEM_MANAGER.add_file_to_global_helper(
    file_path="/path/to/file.txt",
    description="Custom description of the file",
    origin="custom_script",
    llm_summarizer=None,  # Skip LLM generation
)

# Add with LLM-generated description
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

FILESYSTEM_MANAGER.add_file_to_global_helper(
    file_path="/path/to/file.txt",
    llm_summarizer=llm,  # Will generate description
    origin="script_scan",
)
```

### Retrieving File Information

#### Get Description for a Specific File

```python
description = FILESYSTEM_MANAGER.get_file_description_from_helper("/path/to/file.txt")
if description:
    print(f"File description: {description}")
else:
    print("File not found in global helper")
```

#### Get Global Helper as Context String

```python
context = FILESYSTEM_MANAGER.get_global_helper_as_context()
# Returns formatted string suitable for agent context
print(context)
```

Example output:
```
### Global File Helper Catalog:

**config.json**
  - Path: /absolute/path/to/config.json
  - Description: Configuration file with system settings and runtime parameters

**model.py**
  - Path: /absolute/path/to/model.py
  - Description: Defines chatbot model architecture and key operations
```

## Agent Integration

### Automatic Context Inclusion

The global helper is automatically included in every agent interaction:

```python
# In agents.py create_messages()
def create_messages(state):
    messages = [SystemMessage(content=system_prompt)]
    
    # Global helper is ALWAYS included
    global_helper_context = FILESYSTEM_MANAGER.get_global_helper_as_context()
    if global_helper_context and global_helper_context != "No files in global helper catalog.":
        helper_message = HumanMessage(
            content=f"{global_helper_context}\n"
            f"Note: These files are available for reference. Use the appropriate tools to access their content if needed."
        )
        messages.append(helper_message)
    
    # ... rest of message creation
```

### Enhanced File Context

When a file is set as context, its description from the global helper is automatically included:

```python
# In agents.py set_context()
AGENT.set_context("/path/to/file.txt", context_type="file info")

# If file is in global helper, context becomes:
# File: file.txt
# Path: /path/to/file.txt
# Description: [LLM-generated description from helper]
```

## LLM Description Generation

### Prompt Template

File descriptions are generated using the `file_summary_prompt()` from `tools/tool_prompts_texts.py`:

```python
def file_summary_prompt(file_type: str, file_content: str):
    # Generates prompt asking LLM to:
    # 1. Provide brief overview (2-3 sentences)
    # 2. Identify main categories/types of data
    # 3. Highlight key patterns, trends, insights
    # 4. Estimate size/scope of data
    # Max 200 words
```

### Async Operation

Description generation is designed to be non-blocking:
- Occurs when files are added to the helper
- Runs in a background thread to avoid blocking main operations
- File is immediately added with "Generating description..." placeholder
- Description is updated asynchronously once generation completes
- Falls back to "No description available" if LLM is unavailable
- Errors are logged but don't prevent file tracking

## Configuration

### File Locations

- **Global Helper**: `{DATA_FILES_DIR}/global_file_helper.json`
- **Memory Files**: `{FILE_MEMORY_DIR}/*`
- **Data Files**: `{DATA_FILES_DIR}/*`

Default paths (from `general/config.py`):
```python
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_MEMORY_DIR = os.path.join(MAIN_DIR, "memory_files")
DATA_FILES_DIR = os.path.join(MAIN_DIR, "data_files")
```

## Error Handling

The system is designed to gracefully handle errors:

1. **Missing LLM**: Falls back to "No description available"
2. **File Not Found**: Raises `FileNotFoundError` with clear message
3. **Missing Global Helper**: Automatically created on first use
4. **Corrupted JSON**: Reinitializes with empty catalog

Example error handling:
```python
try:
    FILESYSTEM_MANAGER.add_file_to_global_helper(
        file_path=filepath,
        llm_summarizer=llm,
    )
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except Exception as e:
    logger.warning(f"Failed to add to global helper: {e}")
    # Operation continues - global helper is supplementary
```

## Best Practices

1. **Always provide LLM for description generation** when adding files interactively
2. **Use meaningful origin tags** to track file sources
3. **Update descriptions periodically** if file content changes significantly
4. **Clean up stale entries** by checking file existence before operations
5. **Monitor global helper size** to prevent context overflow

## Future Enhancements

Potential improvements for future releases:

1. **Automatic cleanup** of entries for deleted files
2. **Description caching** to avoid redundant LLM calls
3. **Multi-language support** for descriptions
4. **File type-specific** description templates
5. **Version tracking** for file changes
6. **Batch description generation** for multiple files
7. **Description quality scoring** and refinement

## Troubleshooting

### Issue: Files not appearing in global helper

**Solution**: Check that `add_file_to_global_helper()` is being called after file uploads

### Issue: Descriptions showing "No description available"

**Solution**: Ensure LLM is initialized and passed to `add_file_to_global_helper()`

### Issue: Global helper context too large

**Solution**: Consider implementing pagination or selective file inclusion based on relevance

### Issue: Descriptions are outdated

**Solution**: Re-add files with `update_existing=True` to regenerate descriptions

## API Reference

### FileSystemManager Methods

#### `add_file_to_global_helper(file_path, description="", origin="user_upload", llm_summarizer=None)`
Adds a file to the global helper catalog.

**Parameters:**
- `file_path` (str): Full path to the file
- `description` (str, optional): Pre-defined description
- `origin` (str, optional): Source of the file entry
- `llm_summarizer` (BaseChatModel, optional): LLM for description generation

**Returns:** `self` for method chaining

**Raises:** `FileNotFoundError` if file doesn't exist

#### `get_file_description_from_helper(file_path)`
Retrieves a file's description from the global helper.

**Parameters:**
- `file_path` (str): Path to the file (relative or absolute)

**Returns:** `str` description if found, `None` otherwise

#### `get_global_helper_as_context()`
Gets the global file helper as a formatted string for agent context.

**Returns:** `str` formatted context string

## Example Workflows

### Workflow 1: User Uploads File

1. User uploads `report.pdf` through UI
2. `save_to_memory_files()` saves file to `memory_files/`
3. `add_file_to_global_helper()` is called with LLM
4. LLM generates description: "Annual report with financial data..."
5. Entry added to `global_file_helper.json`
6. File context set with description
7. Agent receives enhanced context in next query

### Workflow 2: Agent Queries About Files

1. User asks: "What files do I have uploaded?"
2. Agent receives global helper context automatically
3. Agent responds with list from global helper catalog
4. Descriptions help user understand file purposes
5. User can request specific file content using tools

### Workflow 3: Batch File Processing

```python
import os
from tools.tool_register import FILESYSTEM_MANAGER
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Process all files in a directory
for filename in os.listdir("/path/to/files"):
    filepath = os.path.join("/path/to/files", filename)
    if os.path.isfile(filepath):
        try:
            FILESYSTEM_MANAGER.add_file_to_global_helper(
                file_path=filepath,
                origin="batch_scan",
                llm_summarizer=llm,
            )
            print(f"✓ Processed {filename}")
        except Exception as e:
            print(f"✗ Failed {filename}: {e}")
```

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-12  
**Author**: Copilot Agent
