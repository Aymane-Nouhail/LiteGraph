"""
Graph orchestration system for workflow execution with edge conditions and parallel branches.
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from .node import Node
from .state import State


class Edge(BaseModel):
    """Represents an edge between two nodes in the graph."""
    
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphConfig(BaseModel):
    """Configuration for graph execution."""
    
    enable_checkpointing: bool = False
    checkpoint_dir: Optional[str] = None
    auto_save_checkpoints: bool = True
    enable_parallel_branches: bool = True
    max_parallel_nodes: int = 10
    enable_loop_prevention: bool = True
    max_iterations: int = 1000
    enable_metrics: bool = True


class Graph:
    """
    Directed graph for workflow orchestration with state management.
    
    Supports conditional edges, parallel execution, loop prevention, and checkpointing.
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.start_nodes: Set[str] = set()
        self.execution_history: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        
        # Execution tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = 0.0
    
    def add_node(self, node: Union[Node, Callable], start: bool = False) -> str:
        """
        Add a node to the graph.
        
        Args:
            node: Node instance or function to wrap in a Node
            start: Whether this is a start node
        
        Returns:
            Node name
        """
        if isinstance(node, Callable) and not isinstance(node, Node):
            # If it's a function, check if it has a _node attribute (from @node decorator)
            if hasattr(node, '_node'):
                node = node._node
            else:
                # Create a basic node
                node = Node(node)
        
        node_name = node.name
        self.nodes[node_name] = node
        
        if start or node.is_start:
            self.start_nodes.add(node_name)
        
        return node_name
    
    def add_edge(
        self, 
        from_node: str, 
        to_node: str, 
        condition: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            from_node: Source node name
            to_node: Target node name
            condition: Optional condition function that takes state and returns bool
            metadata: Optional metadata for the edge
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found in graph")
        if to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found in graph")
        
        edge = Edge(
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            metadata=metadata or {}
        )
        self.edges.append(edge)
    
    def get_successors(self, node_name: str, state: State) -> List[str]:
        """Get successor nodes that can be executed based on edge conditions."""
        successors = []
        for edge in self.edges:
            if edge.from_node == node_name:
                if edge.condition is None or edge.condition(state):
                    successors.append(edge.to_node)
        return successors
    
    def get_predecessors(self, node_name: str) -> List[str]:
        """Get predecessor nodes."""
        predecessors = []
        for edge in self.edges:
            if edge.to_node == node_name:
                predecessors.append(edge.from_node)
        return predecessors
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node_name: str, path: List[str]) -> None:
            if node_name in rec_stack:
                # Found a cycle
                cycle_start = path.index(node_name)
                cycles.append(path[cycle_start:] + [node_name])
                return
            
            if node_name in visited:
                return
            
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)
            
            for edge in self.edges:
                if edge.from_node == node_name:
                    dfs(edge.to_node, path.copy())
            
            rec_stack.remove(node_name)
        
        for node_name in self.nodes:
            if node_name not in visited:
                dfs(node_name, [])
        
        return cycles
    
    def validate(self) -> None:
        """Validate the graph structure."""
        if not self.start_nodes:
            raise ValueError("Graph must have at least one start node")
        
        if self.config.enable_loop_prevention:
            cycles = self._detect_cycles()
            if cycles:
                raise ValueError(f"Graph contains cycles: {cycles}")
        
        # Check for unreachable nodes
        reachable = set(self.start_nodes)
        to_visit = deque(self.start_nodes)
        
        while to_visit:
            current = to_visit.popleft()
            for edge in self.edges:
                if edge.from_node == current:
                    if edge.to_node not in reachable:
                        reachable.add(edge.to_node)
                        to_visit.append(edge.to_node)
        
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            print(f"Warning: Unreachable nodes detected: {unreachable}")
    
    def to_dot(self) -> str:
        """Export graph to DOT format for visualization."""
        lines = ["digraph G {"]
        
        # Add nodes
        for node_name, node in self.nodes.items():
            shape = "doublecircle" if node_name in self.start_nodes else "circle"
            lines.append(f'  "{node_name}" [shape={shape}, label="{node_name}"];')
        
        # Add edges
        for edge in self.edges:
            label = ""
            if edge.condition:
                label = f' [label="condition"]'
            lines.append(f'  "{edge.from_node}" -> "{edge.to_node}"{label};')
        
        lines.append("}")
        return "\n".join(lines)
    
    async def run(
        self, 
        initial_state: Optional[Union[Dict[str, Any], State]] = None,
        start_nodes: Optional[List[str]] = None
    ) -> State:
        """
        Execute the graph workflow.
        
        Args:
            initial_state: Initial state data or State instance
            start_nodes: Optional list of start nodes (overrides graph start nodes)
        
        Returns:
            Final state after execution
        """
        start_time = time.time()
        self._execution_count += 1
        
        # Validate graph
        self.validate()
        
        # Initialize state
        if isinstance(initial_state, dict):
            state = State(data=initial_state)
        elif isinstance(initial_state, State):
            state = initial_state
        else:
            state = State()
        
        # Enable checkpointing if configured
        if self.config.enable_checkpointing and self.config.checkpoint_dir:
            state.enable_checkpointing(
                self.config.checkpoint_dir, 
                self.config.auto_save_checkpoints
            )
        
        # Determine start nodes
        execution_nodes = start_nodes or list(self.start_nodes)
        if not execution_nodes:
            raise ValueError("No start nodes specified")
        
        # Track execution
        executed_nodes: Set[str] = set()
        execution_order: List[str] = []
        iteration_count = 0
        
        # Main execution loop
        while execution_nodes and iteration_count < self.config.max_iterations:
            iteration_count += 1
            
            # Execute nodes in parallel if enabled
            if self.config.enable_parallel_branches and len(execution_nodes) > 1:
                # Limit parallel execution
                current_batch = list(execution_nodes)[:self.config.max_parallel_nodes]
                tasks = [self.nodes[node_name].execute(state) for node_name in current_batch]
                await asyncio.gather(*tasks)
                
                for node_name in current_batch:
                    executed_nodes.add(node_name)
                    execution_order.append(node_name)
                    execution_nodes.remove(node_name)
            else:
                # Sequential execution
                node_name = execution_nodes.pop(0)
                await self.nodes[node_name].execute(state)
                executed_nodes.add(node_name)
                execution_order.append(node_name)
            
            # Find next nodes to execute
            next_nodes = set()
            for node_name in executed_nodes:
                successors = self.get_successors(node_name, state)
                for successor in successors:
                    # Check if all predecessors are executed
                    predecessors = self.get_predecessors(successor)
                    if all(pred in executed_nodes for pred in predecessors):
                        next_nodes.add(successor)
            
            # Add new nodes to execution queue
            for node_name in next_nodes:
                if node_name not in executed_nodes and node_name not in execution_nodes:
                    execution_nodes.append(node_name)
        
        if iteration_count >= self.config.max_iterations:
            raise RuntimeError(f"Graph execution exceeded maximum iterations ({self.config.max_iterations})")
        
        # Update metrics
        self._last_execution_time = time.time() - start_time
        self._total_execution_time += self._last_execution_time
        
        # Record execution history
        execution_record = {
            "timestamp": time.time(),
            "execution_time": self._last_execution_time,
            "execution_order": execution_order,
            "final_state": state.to_dict(),
        }
        self.execution_history.append(execution_record)
        
        # Update metrics
        if self.config.enable_metrics:
            self.metrics = {
                "total_executions": self._execution_count,
                "total_execution_time": self._total_execution_time,
                "average_execution_time": self._total_execution_time / self._execution_count,
                "last_execution_time": self._last_execution_time,
                "node_metrics": {
                    name: node.get_metrics() 
                    for name, node in self.nodes.items()
                }
            }
        
        return state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for the graph."""
        return self.metrics.copy()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history.copy() 