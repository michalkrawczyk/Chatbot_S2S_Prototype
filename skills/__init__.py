"""
Skills Module - Modular Agent Capabilities

This module provides a skills-based architecture for the agent, enabling
dynamic discovery, registration, and composition of capabilities.

Architecture Overview:
- BaseSkill: Abstract base class defining the skill interface
- SkillRegistry: Central registry for skill discovery and management
- Concrete Skills: Specific implementations (file_management_skill, etc.)

Reference: Claude Skills Methodology
https://code.claude.com/docs/en/skills

Usage:
    from skills import get_registry
    
    # Get the global registry
    registry = get_registry()
    
    # Discover and register all skills
    registry.discover_skills()
    
    # Get all tools for the agent
    tools = registry.get_all_tools()
"""

from skills.base_skill import BaseSkill, SkillMetadata
from skills.skill_registry import SkillRegistry, get_registry


__all__ = [
    "BaseSkill",
    "SkillMetadata",
    "SkillRegistry",
    "get_registry",
]
