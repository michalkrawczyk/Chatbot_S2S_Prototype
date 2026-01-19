# Skills Architecture Guide

## Overview

This guide explains how to work with the skills-based modular architecture for the chatbot agent. The skills system enables dynamic discovery, registration, and composition of agent capabilities.

**Reference:** [Claude Skills Methodology](https://code.claude.com/docs/en/skills)

## What are Skills?

Skills are modular, self-contained units of functionality that provide specific capabilities to the agent. Each skill:

- **Single Responsibility**: Handles one specific capability (e.g., file management, data analysis)
- **Isolation**: Self-contained with minimal dependencies
- **Discoverability**: Can be automatically discovered and registered
- **Documented**: Includes YAML frontmatter and clear usage examples
- **Error Handling**: Robust error handling with graceful degradation

## Architecture Components

### 1. BaseSkill Class

All skills inherit from `BaseSkill`, which provides:
- Metadata definition (name, version, description, tags)
- Tool registration interface
- Validation and lifecycle management
- Standard error handling patterns

### 2. SkillRegistry

The central registry manages:
- Skill discovery and registration
- Tool aggregation from all skills
- Tag-based skill retrieval
- Lifecycle management (initialization, cleanup)

### 3. Individual Skills

Concrete skill implementations in the `skills/` directory:
- `file_management_skill.py` - File system operations
- `datasheet_management_skill.py` - CSV/Excel data operations
- `google_drive_skill.py` - Google Drive integration

## Creating a New Skill

### Step 1: Create the Skill File

Create a new file in `skills/` with the naming convention `{name}_skill.py`:

```python
"""
---
name: my_skill
version: 1.0.0
description: |
  Brief description of what this skill does
author: Your Name
tags:
  - category1
  - category2
created_at: 2026-01-19
reference: https://code.claude.com/docs/en/skills
---

My Skill Documentation

Detailed description of the skill's purpose, capabilities, and usage.

Usage Example:
    from skills.my_skill import MySkill
    
    skill = MySkill()
    tools = skill.tools
"""

from typing import List, Any
from langchain_core.tools import tool

from skills.base_skill import BaseSkill, SkillMetadata


class MySkill(BaseSkill):
    """
    Brief description of the skill.
    
    Capabilities:
    - Capability 1
    - Capability 2
    
    Error Handling:
    - Error scenario 1
    - Error scenario 2
    """
    
    def __init__(self):
        """Initialize the skill."""
        super().__init__()
    
    def _define_metadata(self) -> SkillMetadata:
        """Define metadata for the skill."""
        return SkillMetadata(
            name="my_skill",
            version="1.0.0",
            description="Brief description",
            author="Your Name",
            tags=["category1", "category2"],
        )
    
    def _register_tools(self) -> List[Any]:
        """Register skill tools."""
        
        @tool
        def my_tool(param: str) -> str:
            """
            Tool description.
            
            Args:
                param: Parameter description
            
            Returns:
                Result description
                
            Example:
                >>> my_tool("test")
                'result'
            """
            try:
                # Implementation here
                return f"Result: {param}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        return [my_tool]
```

### Step 2: Test the Skill

Create a simple test to verify your skill works:

```python
from skills.my_skill import MySkill

# Create skill instance
skill = MySkill()

# Check metadata
print(f"Skill: {skill.name} v{skill.version}")
print(f"Description: {skill.description}")
print(f"Tools: {len(skill.tools)}")

# Test tools
for tool in skill.tools:
    print(f"  - {tool.name}")
```

### Step 3: Register with the Agent

The skill will be automatically discovered if it follows the naming convention `*_skill.py`. Otherwise, manually register it:

```python
from skills import get_registry
from skills.my_skill import MySkill

registry = get_registry()
skill = MySkill()
registry.register_skill(skill)
```

## Using Skills in the Agent

### Automatic Discovery

Skills are automatically discovered and registered when the agent initializes:

```python
from skills import get_registry

registry = get_registry()
registry.discover_skills("skills")  # Scans skills/ directory

print(f"Discovered {len(registry)} skills")
```

### Manual Registration

You can manually register skills if needed:

```python
from skills import get_registry
from skills.my_skill import MySkill

registry = get_registry()
skill = MySkill()
registry.register_skill(skill)
```

### Getting Tools for Agent

```python
from skills import get_registry

registry = get_registry()
registry.discover_skills()

# Get all tools from all skills
all_tools = registry.get_all_tools()

# Get tools as dictionary
tools_dict = registry.get_tools_dict()

# Use with LangChain agent
llm_with_tools = llm.bind_tools(all_tools)
```

## Skill Best Practices

### 1. Documentation

Every skill must include:
- YAML frontmatter with metadata
- Clear docstring explaining purpose
- Usage examples
- Error handling documentation

### 2. Error Handling

All tool functions should:
- Catch exceptions and return error messages (not raise)
- Log errors for debugging
- Provide helpful error messages to users

```python
@tool
def my_tool(param: str) -> str:
    """Tool description."""
    try:
        # Implementation
        result = process(param)
        return result
    except FileNotFoundError:
        return f"Error: File '{param}' not found"
    except Exception as e:
        logger.error(f"Error in my_tool: {e}")
        return f"Error: {str(e)}"
```

### 3. Single Responsibility

Each skill should focus on one specific capability:
- ✅ Good: `FileManagementSkill` handles file operations
- ✅ Good: `DatasheetSkill` handles CSV/Excel operations
- ❌ Bad: `UtilitySkill` handles files, data, and API calls

### 4. Minimal Dependencies

Keep skills self-contained:
- Import only what you need
- Use standard library when possible
- Document external dependencies

### 5. Performance

Consider token limits and performance:
- Use chunked access for large data
- Provide summary operations
- Avoid loading entire files into context

## Advanced Features

### Skill Configuration

Skills can accept configuration:

```python
class MySkill(BaseSkill):
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with configuration."""
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint", "default")
        return True
```

### Skill Cleanup

Implement cleanup for resources:

```python
class MySkill(BaseSkill):
    def cleanup(self):
        """Cleanup skill resources."""
        if hasattr(self, 'connection'):
            self.connection.close()
```

### Tag-Based Retrieval

Retrieve skills by category:

```python
registry = get_registry()

# Get all data-related skills
data_skills = registry.get_skills_by_tag("data")

# Get all cloud integration skills
cloud_skills = registry.get_skills_by_tag("cloud")
```

## Debugging Skills

### List All Skills

```python
from skills import get_registry

registry = get_registry()
skills_info = registry.list_skills()

for skill in skills_info:
    print(f"- {skill['name']} v{skill['version']}")
    print(f"  {skill['description']}")
    print(f"  Tags: {', '.join(skill['tags'])}")
    print(f"  Tools: {skill['tool_count']}")
```

### Check Skill Tools

```python
registry = get_registry()
skill = registry.get_skill("file_management")

if skill:
    capabilities = skill.get_capabilities()
    print(f"Skill: {capabilities['name']}")
    print("Tools:")
    for tool in capabilities['tools']:
        print(f"  - {tool['name']}: {tool['description']}")
```

## Migration from Legacy Tools

### Before (Legacy)

```python
from tools import DEFINED_TOOLS

llm_with_tools = llm.bind_tools(DEFINED_TOOLS)
```

### After (Skills-Based)

```python
from skills import get_registry

registry = get_registry()
registry.discover_skills()
tools = registry.get_all_tools()

llm_with_tools = llm.bind_tools(tools)
```

## Troubleshooting

### Skills Not Discovered

**Problem**: Skills are not being automatically discovered.

**Solution**:
1. Ensure skill files follow naming convention `*_skill.py`
2. Ensure skills are in the `skills/` directory
3. Check that skills inherit from `BaseSkill`
4. Verify no syntax errors in skill files

### Tool Not Found

**Problem**: Agent reports "Tool 'xyz' not found".

**Solution**:
1. Check skill is registered: `registry.list_skills()`
2. Verify tool is in skill's `_register_tools()` method
3. Ensure tool name matches what agent is requesting

### Import Errors

**Problem**: Import errors when loading skills.

**Solution**:
1. Check all dependencies are installed
2. Verify import paths are correct
3. Ensure `skills/__init__.py` exists

## Additional Resources

- [Claude Skills Methodology](https://code.claude.com/docs/en/skills) - Official documentation
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/) - Tool creation guide
- [BaseSkill API](skills/base_skill.py) - Base skill implementation
- [SkillRegistry API](skills/skill_registry.py) - Registry implementation

## Support

For questions or issues with the skills system:
1. Check existing skill implementations for examples
2. Review the troubleshooting section above
3. Check logs for error messages
4. Consult the base skill and registry source code

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-19  
**Author**: Chatbot Development Team
