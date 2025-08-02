"""
State management for graph workflows with persistence and checkpointing.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field


class State(BaseModel):
    """
    Mutable state container for graph workflows with persistence capabilities.
    
    Supports checkpointing, versioning, and optional on-disk storage.
    """
    
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._checkpoint_dir: Optional[Path] = None
        self._auto_save: bool = False
    
    def __getitem__(self, key: str) -> Any:
        """Get value from state data."""
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value in state data and update timestamp."""
        self.data[key] = value
        self.updated_at = time.time()
        self.version += 1
        
        if self._auto_save and self._checkpoint_dir:
            self.save_checkpoint()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in state data."""
        return key in self.data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in state data."""
        self[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update state with multiple key-value pairs."""
        for key, value in data.items():
            self[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "data": self.data,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create state from dictionary."""
        return cls(**data)
    
    def enable_checkpointing(self, checkpoint_dir: Union[str, Path], auto_save: bool = True) -> None:
        """Enable checkpointing to specified directory."""
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._auto_save = auto_save
    
    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        """Save current state as checkpoint."""
        if not self._checkpoint_dir:
            raise ValueError("Checkpointing not enabled. Call enable_checkpointing() first.")
        
        if name is None:
            name = f"checkpoint_v{self.version}_{int(self.updated_at)}.json"
        
        checkpoint_path = self._checkpoint_dir / name
        with open(checkpoint_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return checkpoint_path
    
    def load_checkpoint(self, name: str) -> None:
        """Load state from checkpoint."""
        if not self._checkpoint_dir:
            raise ValueError("Checkpointing not enabled. Call enable_checkpointing() first.")
        
        checkpoint_path = self._checkpoint_dir / name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # Update current state
        self.data = data["data"]
        self.metadata = data["metadata"]
        self.version = data["version"]
        self.created_at = data["created_at"]
        self.updated_at = data["updated_at"]
    
    def list_checkpoints(self) -> list[str]:
        """List available checkpoint files."""
        if not self._checkpoint_dir:
            return []
        
        return [f.name for f in self._checkpoint_dir.glob("checkpoint_*.json")]
    
    def clear(self) -> None:
        """Clear all data from state."""
        self.data.clear()
        self.metadata.clear()
        self.version = 1
        self.updated_at = time.time() 