"""
Skill Registry for Dynamic Skill Discovery and Management

This module provides centralized registration and discovery of agent skills,
enabling the agent to dynamically compose capabilities at runtime.

Reference: Claude Skills Methodology for dynamic skill composition
https://code.claude.com/docs/en/skills
"""

import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import importlib
import inspect

from skills.base_skill import BaseSkill


logger = logging.getLogger("SkillRegistry")


class SkillRegistry:
    """
    Central registry for agent skills.
    
    The registry manages skill lifecycle including:
    - Discovery: Automatic detection of available skills
    - Registration: Adding skills to the available pool
    - Retrieval: Getting skills by name or tag
    - Initialization: Setting up skills with configuration
    
    This enables dynamic skill composition where the agent can utilize
    skills based on task requirements without hardcoded dependencies.
    
    Reference: Claude Skills Methodology - Dynamic Skill Discovery
    https://code.claude.com/docs/en/skills#discovery
    """
    
    def __init__(self):
        """Initialize the skill registry."""
        self._skills: Dict[str, BaseSkill] = {}
        self._skill_classes: Dict[str, Type[BaseSkill]] = {}
        self._tags_index: Dict[str, List[str]] = {}
        logger.info("Skill registry initialized")
    
    def register_skill(self, skill: BaseSkill, override: bool = False) -> bool:
        """
        Register a skill instance.
        
        Args:
            skill: Skill instance to register
            override: If True, allows overriding existing skill with same name
            
        Returns:
            True if registration successful, False otherwise
            
        Raises:
            ValueError: If skill with same name exists and override=False
        """
        try:
            skill_name = skill.name
            
            # Check for existing skill
            if skill_name in self._skills and not override:
                raise ValueError(
                    f"Skill '{skill_name}' already registered. "
                    f"Use override=True to replace."
                )
            
            # Register the skill
            self._skills[skill_name] = skill
            self._skill_classes[skill_name] = skill.__class__
            
            # Update tags index
            for tag in skill.metadata.tags:
                if tag not in self._tags_index:
                    self._tags_index[tag] = []
                if skill_name not in self._tags_index[tag]:
                    self._tags_index[tag].append(skill_name)
            
            logger.info(
                f"Registered skill: {skill_name} v{skill.version} "
                f"with {len(skill.tools)} tool(s)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to register skill {skill.name}: {e}")
            return False
    
    def register_skill_class(
        self, 
        skill_class: Type[BaseSkill],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a skill by class, instantiating it automatically.
        
        Args:
            skill_class: Skill class to instantiate and register
            config: Optional configuration for skill initialization
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Instantiate the skill
            skill_instance = skill_class()
            
            # Initialize with config if provided
            if config:
                skill_instance.initialize(config)
            
            # Register the instance
            return self.register_skill(skill_instance)
            
        except Exception as e:
            logger.error(f"Failed to register skill class {skill_class.__name__}: {e}")
            return False
    
    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """
        Retrieve a skill by name.
        
        Args:
            name: Name of the skill to retrieve
            
        Returns:
            Skill instance if found, None otherwise
        """
        return self._skills.get(name)
    
    def get_skills_by_tag(self, tag: str) -> List[BaseSkill]:
        """
        Retrieve all skills with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of skills with the specified tag
        """
        skill_names = self._tags_index.get(tag, [])
        return [self._skills[name] for name in skill_names if name in self._skills]
    
    def get_all_skills(self) -> List[BaseSkill]:
        """
        Get all registered skills.
        
        Returns:
            List of all registered skill instances
        """
        return list(self._skills.values())
    
    def get_all_tools(self) -> List[Any]:
        """
        Get all tools from all registered skills.
        
        Returns:
            Flattened list of all tools from all skills
        """
        tools = []
        for skill in self._skills.values():
            tools.extend(skill.tools)
        return tools
    
    def get_tools_dict(self) -> Dict[str, Any]:
        """
        Get all tools as a dictionary keyed by tool name.
        
        Returns:
            Dictionary mapping tool names to tool objects
        """
        tools_dict = {}
        for skill in self._skills.values():
            for tool in skill.tools:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                tools_dict[tool_name] = tool
        return tools_dict
    
    def discover_skills(self, skills_dir: str = "skills") -> int:
        """
        Automatically discover and register skills from a directory.
        
        Scans the specified directory for Python modules that define
        BaseSkill subclasses and registers them automatically.
        
        Args:
            skills_dir: Directory to scan for skills (relative to project root)
            
        Returns:
            Number of skills discovered and registered
            
        Reference: Claude Skills Methodology - Automatic Discovery
        https://code.claude.com/docs/en/skills#discovery
        """
        discovered_count = 0
        skills_path = Path(skills_dir)
        
        if not skills_path.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return 0
        
        logger.info(f"Discovering skills in: {skills_dir}")
        
        # Scan for Python files
        for py_file in skills_path.glob("*_skill.py"):
            try:
                # Import the module
                module_name = f"{skills_dir}.{py_file.stem}"
                module = importlib.import_module(module_name)
                
                # Find BaseSkill subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseSkill) and 
                        obj is not BaseSkill and
                        obj.__module__ == module.__name__):
                        
                        # Register the skill class
                        if self.register_skill_class(obj):
                            discovered_count += 1
                            logger.info(f"Discovered and registered: {name}")
                        
            except Exception as e:
                logger.error(f"Error discovering skills in {py_file}: {e}")
        
        logger.info(f"Skill discovery complete: {discovered_count} skill(s) registered")
        return discovered_count
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """
        List all registered skills with their metadata.
        
        Returns:
            List of dictionaries containing skill information
        """
        return [
            {
                "name": skill.name,
                "version": skill.version,
                "description": skill.description,
                "tags": skill.metadata.tags,
                "tool_count": len(skill.tools)
            }
            for skill in self._skills.values()
        ]
    
    def unregister_skill(self, name: str) -> bool:
        """
        Remove a skill from the registry.
        
        Args:
            name: Name of the skill to remove
            
        Returns:
            True if skill was removed, False if not found
        """
        if name not in self._skills:
            logger.warning(f"Skill '{name}' not found in registry")
            return False
        
        skill = self._skills[name]
        
        # Cleanup skill resources
        try:
            skill.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up skill {name}: {e}")
        
        # Remove from registry
        del self._skills[name]
        del self._skill_classes[name]
        
        # Remove from tags index
        for tag in skill.metadata.tags:
            if tag in self._tags_index and name in self._tags_index[tag]:
                self._tags_index[tag].remove(name)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
        
        logger.info(f"Unregistered skill: {name}")
        return True
    
    def clear(self):
        """Remove all skills from the registry."""
        skill_names = list(self._skills.keys())
        for name in skill_names:
            self.unregister_skill(name)
        logger.info("Skill registry cleared")
    
    def __len__(self) -> int:
        """Get the number of registered skills."""
        return len(self._skills)
    
    def __contains__(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills
    
    def __repr__(self) -> str:
        """Developer representation of the registry."""
        return f"<SkillRegistry: {len(self._skills)} skill(s) registered>"


# Global registry instance
_global_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    """
    Get the global skill registry instance.
    
    Returns:
        Global SkillRegistry instance
    """
    return _global_registry
