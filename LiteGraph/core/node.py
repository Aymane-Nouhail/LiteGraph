"""
Node system for graph workflows with decorators, caching, and resilience features.
"""

import asyncio
import functools
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .state import State


class NodeConfig(BaseModel):
    """Configuration for a graph node."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    cache_ttl: Optional[int] = None  # seconds
    max_retries: int = 0
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    timeout: Optional[float] = None
    hooks: Dict[str, List[Callable]] = Field(default_factory=dict)


class Node:
    """Represents a node in the workflow graph."""
    
    def __init__(
        self,
        func: Callable,
        config: Optional[NodeConfig] = None,
        is_start: bool = False
    ):
        self.func = func
        self.config = config or NodeConfig()
        self.is_start = is_start
        self.name = self.config.name or func.__name__
        self.description = self.config.description or func.__doc__ or ""
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Execution tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
    
    def _get_cache_key(self, state: State) -> str:
        """Generate cache key based on function name and state data."""
        # Create a hash of the function name and relevant state data
        # Exclude state version to make cache key stable across executions
        cache_data = {
            "func": self.func.__name__,
            "state_data": state.data
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if self.config.cache_ttl is None:
            return False
        
        if cache_key not in self._cache_timestamps:
            return False
        
        return time.time() - self._cache_timestamps[cache_key] < self.config.cache_ttl
    
    def _call_hooks(self, hook_type: str, state: State, **kwargs) -> None:
        """Call registered hooks for the given type."""
        hooks = self.config.hooks.get(hook_type, [])
        for hook in hooks:
            try:
                hook(state, **kwargs)
            except Exception as e:
                # Log hook errors but don't fail the node
                print(f"Warning: Hook {hook_type} failed: {e}")
    
    async def execute(self, state: State) -> State:
        """Execute the node function with caching, retries, and hooks."""
        start_time = time.time()
        self.execution_count += 1
        
        # Call pre-execution hooks
        self._call_hooks("before", state, node_name=self.name)
        
        # Generate cache key BEFORE function execution (when state is still clean)
        cache_key = None
        if self.config.cache_ttl:
            cache_key = self._get_cache_key(state)
            if self._is_cache_valid(cache_key):
                self._call_hooks("cache_hit", state, node_name=self.name, cache_key=cache_key)
                # Restore cached state
                cached_data = self._cache[cache_key]
                state.data.update(cached_data)
                return state
        
        # Execute with retries
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Call pre-attempt hooks
                self._call_hooks("before_attempt", state, node_name=self.name, attempt=attempt)
                
                # Execute function with timeout
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        self.func(state), 
                        timeout=self.config.timeout
                    )
                else:
                    result = await self.func(state)
                
                # Update state with result
                if isinstance(result, dict):
                    state.update(result)
                elif result is not None:
                    # If result is not dict, store it in a default key
                    state[f"{self.name}_result"] = result
                
                # Cache result if TTL is set (use the pre-generated cache key)
                if self.config.cache_ttl and cache_key:
                    self._cache[cache_key] = state.data.copy()
                    self._cache_timestamps[cache_key] = time.time()
                
                # Call post-execution hooks
                self.last_execution_time = time.time() - start_time
                self.total_execution_time += self.last_execution_time
                
                self._call_hooks("after", state, node_name=self.name, execution_time=self.last_execution_time)
                
                return state
                
            except Exception as e:
                last_exception = e
                self._call_hooks("error", state, node_name=self.name, error=e, attempt=attempt)
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)
                else:
                    break
        
        # All retries exhausted
        raise last_exception or RuntimeError(f"Node {self.name} failed after {self.config.max_retries + 1} attempts")
    
    def add_hook(self, hook_type: str, hook_func: Callable) -> None:
        """Add a hook function for the specified type."""
        if hook_type not in self.config.hooks:
            self.config.hooks[hook_type] = []
        self.config.hooks[hook_type].append(hook_func)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this node."""
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "last_execution_time": self.last_execution_time,
            "average_execution_time": (
                self.total_execution_time / self.execution_count 
                if self.execution_count > 0 else 0
            ),
            "cache_hit_rate": len(self._cache) / max(self.execution_count, 1),
        }


def node(
    name: Optional[str] = None,
    description: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    timeout: Optional[float] = None,
    is_start: bool = False,
    hooks: Optional[Dict[str, List[Callable]]] = None,
) -> Callable:
    """
    Decorator to mark a function as a graph node.
    
    Args:
        name: Optional name for the node (defaults to function name)
        description: Optional description for the node
        cache_ttl: Cache TTL in seconds (None to disable caching)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries
        retry_backoff: Exponential backoff multiplier
        timeout: Execution timeout in seconds
        is_start: Whether this is a start node
        hooks: Dictionary of hook functions by type
    
    Returns:
        Decorated function that returns a Node instance
    """
    def decorator(func: Callable) -> Node:
        config = NodeConfig(
            name=name,
            description=description,
            cache_ttl=cache_ttl,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            timeout=timeout,
            hooks=hooks or {},
        )
        
        node_instance = Node(func, config, is_start)
        
        # Store the node instance on the function for easy access
        func._node = node_instance
        
        return node_instance
    
    return decorator 