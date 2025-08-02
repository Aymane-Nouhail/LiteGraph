"""
Tool registry for managing and executing agent tools.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Tool(BaseModel):
    """Represents a tool that can be called by an agent."""
    
    name: str
    description: str
    func: Callable
    is_async: bool = False
    parameters: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Determine if function is async
        self.is_async = asyncio.iscoroutinefunction(self.func)
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if self.is_async:
            return await self.func(**kwargs)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.func(**kwargs))


class ToolRegistry:
    """
    Registry for managing tools that can be used by agents.
    
    Provides tool registration, execution, and metadata management.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._execution_count = 0
        self._execution_history: List[Dict[str, Any]] = []
    
    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to register a function as a tool.
        
        Args:
            name: Tool name (defaults to function name)
            description: Tool description (required)
            parameters: Optional parameter schema
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            # If no explicit description was passed, pull from the func docstring
            tool_desc = description or inspect.getdoc(func) or ""
            if not tool_desc.strip():
                raise ValueError(
                    f"Tool '{func.__name__}' requires a description, "
                    "either via the decorator or in the function docstring"
                )

            tool_name = name or func.__name__
            if tool_name in self.tools:
                raise ValueError(f"Tool '{tool_name}' already registered")

            tool = Tool(
                name=tool_name,
                description=tool_desc,
                func=func,
                parameters=parameters
            )
            self.tools[tool_name] = tool
            func._tool = tool
            return func

        return decorator
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "is_async": tool.is_async,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name with given arguments.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        start_time = asyncio.get_event_loop().time()
        self._execution_count += 1
        
        try:
            result = await tool.execute(**kwargs)
            
            # Record execution
            execution_record = {
                "tool_name": name,
                "arguments": kwargs,
                "result": result,
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "success": True
            }
            self._execution_history.append(execution_record)
            
            return result
            
        except Exception as e:
            # Record failed execution
            execution_record = {
                "tool_name": name,
                "arguments": kwargs,
                "error": str(e),
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "success": False
            }
            self._execution_history.append(execution_record)
            
            raise
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get tool execution metrics."""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        successful = sum(1 for record in self._execution_history if record["success"])
        total_time = sum(record["execution_time"] for record in self._execution_history)
        
        return {
            "total_executions": self._execution_count,
            "success_rate": successful / len(self._execution_history),
            "average_execution_time": total_time / len(self._execution_history),
            "tool_usage": self._get_tool_usage_stats()
        }
    
    def _get_tool_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for each tool."""
        usage = {}
        for record in self._execution_history:
            tool_name = record["tool_name"]
            usage[tool_name] = usage.get(tool_name, 0) + 1
        return usage
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
        self._execution_count = 0


# Global tool registry instance
_global_registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to register a function as a tool in the global registry.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (required)
        parameters: Optional parameter schema
    
    Returns:
        Decorated function
    """
    return _global_registry.register(name, description, parameters)


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry 