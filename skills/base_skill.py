"""
Base Skill Class for Modular Agent Architecture

This module defines the base class for all agent skills, following a modular
and extensible architecture pattern. Each skill represents a discrete capability
that the agent can utilize dynamically at runtime.

Reference: Claude Skills Methodology for modular agent design
https://code.claude.com/docs/en/skills
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SkillMetadata:
    """
    Metadata descriptor for a skill.
    
    Attributes:
        name: Unique identifier for the skill
        version: Semantic version string (e.g., "1.0.0")
        description: Brief description of skill's purpose
        author: Skill creator/maintainer
        tags: Categorization tags for discovery
        created_at: ISO timestamp of creation
        updated_at: ISO timestamp of last update
    """
    name: str
    version: str
    description: str
    author: str = "Unknown"
    tags: List[str] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()


class BaseSkill(ABC):
    """
    Abstract base class for all agent skills.
    
    Skills are modular, reusable units of functionality that provide specific
    capabilities to the agent. Each skill follows a standard interface enabling
    dynamic discovery, registration, and execution.
    
    Design Principles:
    1. Single Responsibility: Each skill handles one specific capability
    2. Isolation: Skills are self-contained with minimal dependencies
    3. Discoverability: Metadata enables runtime discovery
    4. Error Handling: Robust error handling with graceful degradation
    5. Documentation: Clear usage examples and API documentation
    
    Reference: Claude Skills Methodology
    https://code.claude.com/docs/en/skills
    """
    
    def __init__(self):
        """Initialize the skill with metadata."""
        self._metadata = self._define_metadata()
        self._tools = self._register_tools()
        self._validate_skill()
    
    @abstractmethod
    def _define_metadata(self) -> SkillMetadata:
        """
        Define the skill's metadata.
        
        This method must be implemented by each concrete skill to provide
        identifying information and categorization.
        
        Returns:
            SkillMetadata instance with skill details
        """
        pass
    
    @abstractmethod
    def _register_tools(self) -> List[Any]:
        """
        Register the tools this skill provides.
        
        Tools are the concrete functions that implement the skill's capabilities.
        They should be LangChain-compatible tool objects.
        
        Returns:
            List of tool objects that implement this skill
        """
        pass
    
    def _validate_skill(self):
        """
        Validate skill configuration.
        
        Ensures that the skill is properly configured with required metadata
        and at least one tool implementation.
        
        Raises:
            ValueError: If skill configuration is invalid
        """
        if not self._metadata:
            raise ValueError(f"Skill {self.__class__.__name__} must define metadata")
        if not self._metadata.name:
            raise ValueError(f"Skill {self.__class__.__name__} must have a name")
        if not self._tools:
            raise ValueError(f"Skill {self._metadata.name} must register at least one tool")
    
    @property
    def metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return self._metadata
    
    @property
    def tools(self) -> List[Any]:
        """Get skill tools."""
        return self._tools
    
    @property
    def name(self) -> str:
        """Get skill name."""
        return self._metadata.name
    
    @property
    def version(self) -> str:
        """Get skill version."""
        return self._metadata.version
    
    @property
    def description(self) -> str:
        """Get skill description."""
        return self._metadata.description
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get a description of this skill's capabilities.
        
        Returns:
            Dictionary containing skill metadata and tool descriptions
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self._metadata.tags,
            "tools": [
                {
                    "name": tool.name if hasattr(tool, 'name') else str(tool),
                    "description": tool.description if hasattr(tool, 'description') else "No description"
                }
                for tool in self._tools
            ]
        }
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the skill with optional configuration.
        
        This method can be overridden by subclasses to perform setup operations
        such as loading resources, establishing connections, or validating
        configuration parameters.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        return True
    
    def cleanup(self):
        """
        Cleanup skill resources.
        
        This method can be overridden by subclasses to perform cleanup operations
        such as closing connections or releasing resources.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the skill."""
        return f"Skill({self.name} v{self.version})"
    
    def __repr__(self) -> str:
        """Developer representation of the skill."""
        return f"<{self.__class__.__name__}: {self.name} v{self.version} with {len(self._tools)} tools>"
