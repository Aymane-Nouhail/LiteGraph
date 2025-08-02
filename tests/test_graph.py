"""
Tests for graph orchestration system.
"""

import pytest
import asyncio
from LiteGraph import Graph, node, State


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    
    @node(description="Start node")
    async def start(state: State) -> dict:
        state["value"] = 1
        return {"step": "start"}
    
    @node(description="Process node")
    async def process(state: State) -> dict:
        state["value"] *= 2
        return {"step": "process"}
    
    @node(description="End node")
    async def end(state: State) -> dict:
        state["value"] += 1
        return {"step": "end"}
    
    graph = Graph()
    graph.add_node(start, start=True)
    graph.add_node(process)
    graph.add_node(end)
    
    graph.add_edge("start", "process")
    graph.add_edge("process", "end")
    
    return graph


@pytest.mark.asyncio
async def test_simple_graph_execution(simple_graph):
    """Test basic graph execution."""
    state = await simple_graph.run()
    
    assert state["value"] == 3  # 1 * 2 + 1
    assert state["step"] == "end"


@pytest.mark.asyncio
async def test_graph_with_conditional_edges():
    """Test graph with conditional edges."""
    
    @node(description="Start")
    async def start(state: State) -> dict:
        state["value"] = 5
        return {"step": "start"}
    
    @node(description="High path")
    async def high(state: State) -> dict:
        state["path"] = "high"
        return {"step": "high"}
    
    @node(description="Low path")
    async def low(state: State) -> dict:
        state["path"] = "low"
        return {"step": "low"}
    
    def condition_high(state: State) -> bool:
        return state["value"] > 3
    
    def condition_low(state: State) -> bool:
        return state["value"] <= 3
    
    graph = Graph()
    graph.add_node(start, start=True)
    graph.add_node(high)
    graph.add_node(low)
    
    graph.add_edge("start", "high", condition=condition_high)
    graph.add_edge("start", "low", condition=condition_low)
    
    # Test high path
    state = await graph.run()
    assert state["path"] == "high"
    
    # Test low path
    async def start_low(state: State) -> dict:
        state["value"] = 2
        return {"step": "start"}
    
    start_node = graph.nodes["start"]
    start_node.func = start_low
    
    state = await graph.run()
    assert state["path"] == "low"


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test parallel node execution."""
    
    @node(description="Start")
    async def start(state: State) -> dict:
        state["value"] = 1
        return {"step": "start"}
    
    @node(description="Parallel 1")
    async def parallel1(state: State) -> dict:
        await asyncio.sleep(0.1)  # Simulate work
        state["p1"] = state["value"] * 2
        return {"step": "parallel1"}
    
    @node(description="Parallel 2")
    async def parallel2(state: State) -> dict:
        await asyncio.sleep(0.1)  # Simulate work
        state["p2"] = state["value"] * 3
        return {"step": "parallel2"}
    
    @node(description="Merge")
    async def merge(state: State) -> dict:
        state["result"] = state["p1"] + state["p2"]
        return {"step": "merge"}
    
    graph = Graph()
    graph.add_node(start, start=True)
    graph.add_node(parallel1)
    graph.add_node(parallel2)
    graph.add_node(merge)
    
    graph.add_edge("start", "parallel1")
    graph.add_edge("start", "parallel2")
    graph.add_edge("parallel1", "merge")
    graph.add_edge("parallel2", "merge")
    
    state = await graph.run()
    
    assert state["p1"] == 2
    assert state["p2"] == 3
    assert state["result"] == 5


@pytest.mark.asyncio
async def test_node_caching():
    """Test node caching functionality."""
    
    call_count = 0
    
    @node(description="Cached node", cache_ttl=60)
    async def cached_node(state: State) -> dict:
        nonlocal call_count
        call_count += 1
        state["count"] = call_count
        return {"step": "cached"}
    
    graph = Graph()
    graph.add_node(cached_node, start=True)
    
    # First execution
    state1 = await graph.run()
    assert state1["count"] == 1
    assert call_count == 1
    

    
    # Second execution with same state should use cache
    # Use the same graph instance to test caching
    state2 = await graph.run()
    assert state2["count"] == 1  # Should be cached
    assert call_count == 1  # Function should not be called again


@pytest.mark.asyncio
async def test_node_retries():
    """Test node retry functionality."""
    
    call_count = 0
    
    @node(description="Retry node", max_retries=2, retry_delay=0.01)
    async def retry_node(state: State) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # Fail first two times
            raise RuntimeError("Temporary failure")
        state["success"] = True
        return {"step": "retry"}
    
    graph = Graph()
    graph.add_node(retry_node, start=True)
    
    state = await graph.run()
    
    assert state["success"] is True
    assert call_count == 3  # Should have retried twice


@pytest.mark.asyncio
async def test_node_hooks():
    """Test node hooks functionality."""
    
    hook_calls = []
    
    def before_hook(state, **kwargs):
        hook_calls.append(f"before_{kwargs['node_name']}")
    
    def after_hook(state, **kwargs):
        hook_calls.append(f"after_{kwargs['node_name']}")
    
    @node(description="Hooked node", hooks={
        "before": [before_hook],
        "after": [after_hook]
    })
    async def hooked_node(state: State) -> dict:
        state["value"] = 42
        return {"step": "hooked"}
    
    graph = Graph()
    graph.add_node(hooked_node, start=True)
    
    state = await graph.run()
    
    assert state["value"] == 42
    assert hook_calls == ["before_hooked_node", "after_hooked_node"]


@pytest.mark.asyncio
async def test_graph_validation():
    """Test graph validation."""
    
    @node(description="Start")
    async def start(state: State) -> dict:
        return {"step": "start"}
    
    @node(description="End")
    async def end(state: State) -> dict:
        return {"step": "end"}
    
    # Test missing start node
    graph = Graph()
    graph.add_node(start)
    graph.add_node(end)
    graph.add_edge("start", "end")
    
    with pytest.raises(ValueError, match="must have at least one start node"):
        await graph.run()
    
    # Test cycle detection
    graph = Graph()
    graph.add_node(start, start=True)
    graph.add_node(end)
    graph.add_edge("start", "end")
    graph.add_edge("end", "start")  # Creates cycle
    
    with pytest.raises(ValueError, match="contains cycles"):
        await graph.run()


@pytest.mark.asyncio
async def test_graph_metrics():
    """Test graph metrics collection."""
    
    @node(description="Test node")
    async def test_node(state: State) -> dict:
        await asyncio.sleep(0.01)  # Small delay for timing
        state["value"] = 42
        return {"step": "test"}
    
    graph = Graph()
    graph.add_node(test_node, start=True)
    
    state = await graph.run()
    
    metrics = graph.get_metrics()
    
    assert metrics["total_executions"] == 1
    assert metrics["last_execution_time"] > 0
    assert "test_node" in metrics["node_metrics"]
    
    node_metrics = metrics["node_metrics"]["test_node"]
    assert node_metrics["execution_count"] == 1
    assert node_metrics["last_execution_time"] > 0


@pytest.mark.asyncio
async def test_graph_to_dot():
    """Test graph DOT export."""
    
    @node(description="Start")
    async def start(state: State) -> dict:
        return {"step": "start"}
    
    @node(description="End")
    async def end(state: State) -> dict:
        return {"step": "end"}
    
    graph = Graph()
    graph.add_node(start, start=True)
    graph.add_node(end)
    graph.add_edge("start", "end")
    
    dot = graph.to_dot()
    
    assert "digraph G {" in dot
    assert '"start"' in dot
    assert '"end"' in dot
    assert '"start" -> "end"' in dot
    assert "doublecircle" in dot  # Start node shape


@pytest.mark.asyncio
async def test_state_persistence():
    """Test state persistence and checkpointing."""
    
    @node(description="Test node")
    async def test_node(state: State) -> dict:
        state["value"] = 42
        return {"step": "test"}
    
    graph = Graph()
    graph.add_node(test_node, start=True)
    
    # Enable checkpointing
    graph.config.enable_checkpointing = True
    graph.config.checkpoint_dir = "/tmp/LiteGraph_test"
    
    state = await graph.run()
    
    assert state["value"] == 42
    
    # Check that checkpoint was created
    checkpoints = state.list_checkpoints()
    assert len(checkpoints) > 0 