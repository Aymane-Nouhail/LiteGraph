"""
Basic graph workflow example.

Demonstrates:
- Node decorators
- Graph construction
- Conditional edges
- Parallel execution
- State management
"""

import asyncio
from LiteGraph import Graph, node, State


@node(description="Initialize the workflow with input data")
async def start_node(state: State) -> dict:
    """Initialize the workflow."""
    print("🚀 Starting workflow...")
    state["input_data"] = "Hello, LiteGraph!"
    state["step"] = "started"
    return {"message": "Workflow initialized"}


@node(description="Process the input data")
async def process_node(state: State) -> dict:
    """Process the input data."""
    print(f"📝 Processing: {state['input_data']}")
    processed_data = state["input_data"].upper()
    state["processed_data"] = processed_data
    state["step"] = "processed"
    return {"processed": processed_data}


@node(description="Validate the processed data")
async def validate_node(state: State) -> dict:
    """Validate the processed data."""
    print(f"✅ Validating: {state['processed_data']}")
    is_valid = len(state["processed_data"]) > 0
    state["is_valid"] = is_valid
    state["step"] = "validated"
    return {"valid": is_valid}


@node(description="Generate final output")
async def output_node(state: State) -> dict:
    """Generate the final output."""
    print("🎯 Generating output...")
    if state.get("is_valid", False):
        result = f"SUCCESS: {state['processed_data']}"
    else:
        result = "ERROR: Invalid data"
    
    state["result"] = result
    state["step"] = "completed"
    return {"result": result}


@node(description="Log the workflow completion")
async def log_node(state: State) -> dict:
    """Log workflow completion."""
    print(f"📊 Workflow completed: {state['result']}")
    state["logged"] = True
    return {"logged": True}


def condition_valid(state: State) -> bool:
    """Condition: data is valid."""
    return state.get("is_valid", False)


def condition_invalid(state: State) -> bool:
    """Condition: data is invalid."""
    return not state.get("is_valid", True)


async def main():
    """Run the basic graph workflow."""
    print("🧠 LiteGraph - Basic Graph Example")
    print("=" * 50)
    
    # Create graph
    graph = Graph()
    
    # Add nodes
    graph.add_node(start_node, start=True)
    graph.add_node(process_node)
    graph.add_node(validate_node)
    graph.add_node(output_node)
    graph.add_node(log_node)
    
    # Add edges
    graph.add_edge("start_node", "process_node")
    graph.add_edge("process_node", "validate_node")
    graph.add_edge("validate_node", "output_node")
    graph.add_edge("output_node", "log_node")
    
    # Add conditional edge for invalid data
    graph.add_edge("validate_node", "log_node", condition=condition_invalid)
    
    # Run the graph
    print("\n🔄 Executing graph...")
    final_state = await graph.run()
    
    # Display results
    print("\n📋 Final State:")
    print(f"  Input: {final_state.get('input_data')}")
    print(f"  Processed: {final_state.get('processed_data')}")
    print(f"  Valid: {final_state.get('is_valid')}")
    print(f"  Result: {final_state.get('result')}")
    print(f"  Step: {final_state.get('step')}")
    
    # Display metrics
    print("\n📊 Graph Metrics:")
    metrics = graph.get_metrics()
    print(f"  Total executions: {metrics.get('total_executions', 0)}")
    print(f"  Last execution time: {metrics.get('last_execution_time', 0):.3f}s")
    
    # Display node metrics
    print("\n🔧 Node Metrics:")
    node_metrics = metrics.get('node_metrics', {})
    for node_name, node_metric in node_metrics.items():
        print(f"  {node_name}: {node_metric.get('execution_count', 0)} executions, "
              f"{node_metric.get('last_execution_time', 0):.3f}s")
    
    # Export graph visualization
    print("\n🖼️  Graph Visualization (DOT format):")
    print(graph.to_dot())


if __name__ == "__main__":
    asyncio.run(main()) 