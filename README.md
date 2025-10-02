# 🧠 LiteGraph

A minimal, production-ready AI workflow framework with graph orchestration and ReAct agents.

<p align="center">
  <img width="240" alt="LiteGraph Logo" src="https://github.com/user-attachments/assets/7af3d32a-4741-4b1e-aefe-d2bb47690c3e" />
</p>


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Features

- **🚀 Async-first & Lightweight**: Built on `asyncio` with minimal dependencies
- **📊 Graph Orchestration**: Define workflows as directed graphs with conditional edges
- **🤖 ReAct Agents**: Intelligent agents with streaming, tool integration, and graceful exit
- **🔧 Multi-agent Coordination**: Supervisor and conditional handoff tools
- **📈 Observability**: Built-in metrics, tracing, and checkpointing
- **🔌 Pluggable**: Work with any LLM API (OpenAI, Anthropic, Ollama...)

## 🚀 Quick Start

### Installation

```bash
pip install LiteGraph
```

### Basic Graph Workflow

```python
import asyncio
from LiteGraph import Graph, node, State

@node(description="Initialize workflow")
async def start(state: State) -> dict:
    state["input"] = "Hello, LiteGraph!"
    return {"step": "started"}

@node(description="Process data")
async def process(state: State) -> dict:
    state["result"] = state["input"].upper()
    return {"step": "processed"}

# Create and run graph
graph = Graph()
graph.add_node(start, start=True)
graph.add_node(process)
graph.add_edge("start", "process")

final_state = await graph.run()
print(f"Result: {final_state['result']}")
```

### ReAct Agent with Tools

```python
import asyncio
from LiteGraph import ReActAgent, tool, OpenAIClient

# Register tools
@tool(description="Calculate mathematical expressions")
def calculate(expression: str) -> str:
    return f"Result: {eval(expression)}"

@tool(description="Get weather for a location")
async def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 25°C"

# Create agent
llm = OpenAIClient()  # Requires OPENAI_API_KEY
agent = ReActAgent(llm)

# Run agent
answer = await agent.run("What's 15 * 23 and what's the weather in Paris?")
print(f"Answer: {answer}")
```

## 📚 Examples

### Advanced Graph with Conditional Edges

```python
import asyncio
from LiteGraph import Graph, node, State

@node(description="Start")
async def start(state: State) -> dict:
    state["value"] = 42
    return {"step": "start"}

@node(description="High path")
async def high_path(state: State) -> dict:
    state["path"] = "high"
    return {"step": "high"}

@node(description="Low path")
async def low_path(state: State) -> dict:
    state["path"] = "low"
    return {"step": "low"}

# Conditional functions
def is_high(state: State) -> bool:
    return state["value"] > 40

def is_low(state: State) -> bool:
    return state["value"] <= 40

# Build graph
graph = Graph()
graph.add_node(start, start=True)
graph.add_node(high_path)
graph.add_node(low_path)

graph.add_edge("start", "high_path", condition=is_high)
graph.add_edge("start", "low_path", condition=is_low)

# Execute
state = await graph.run()
print(f"Path taken: {state['path']}")
```

### Streaming ReAct Agent

```python
import asyncio
from LiteGraph import ReActAgent, tool, OpenAIClient

@tool(description="Search for information")
async def search(topic: str) -> str:
    return f"Found information about {topic}"

async def streaming_callback(chunk: str, is_complete: bool):
    if is_complete:
        print("\n✅ Done!")
    else:
        print(chunk, end="", flush=True)

# Create agent
llm = OpenAIClient()
agent = ReActAgent(llm)

# Run with streaming
answer = await agent.run_streaming(
    "Search for information about AI and summarize it",
    streaming_callback
)
```

### Parallel Node Execution

```python
import asyncio
from LiteGraph import Graph, node, State

@node(description="Parallel task 1")
async def task1(state: State) -> dict:
    await asyncio.sleep(0.1)
    state["result1"] = "Task 1 completed"
    return {"step": "task1"}

@node(description="Parallel task 2")
async def task2(state: State) -> dict:
    await asyncio.sleep(0.1)
    state["result2"] = "Task 2 completed"
    return {"step": "task2"}

@node(description="Merge results")
async def merge(state: State) -> dict:
    state["final"] = f"{state['result1']} + {state['result2']}"
    return {"step": "merge"}

# Build graph with parallel execution
graph = Graph()
graph.add_node(task1, start=True)
graph.add_node(task2, start=True)
graph.add_node(merge)

graph.add_edge("task1", "merge")
graph.add_edge("task2", "merge")

# Execute (tasks run in parallel)
state = await graph.run()
print(f"Final result: {state['final']}")
```

## 🔧 API Reference

### Core Components

#### `@node` Decorator

```python
@node(
    name: Optional[str] = None,
    description: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    timeout: Optional[float] = None,
    is_start: bool = False,
    hooks: Optional[Dict[str, List[Callable]]] = None,
)
```

#### `Graph` Class

```python
class Graph:
    def add_node(self, node: Union[Node, Callable], start: bool = False) -> str
    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable] = None)
    async def run(self, initial_state: Optional[Union[Dict, State]] = None) -> State
    def to_dot(self) -> str
    def get_metrics(self) -> Dict[str, Any]
```

#### `State` Class

```python
class State:
    def __getitem__(self, key: str) -> Any
    def __setitem__(self, key: str, value: Any) -> None
    def get(self, key: str, default: Any = None) -> Any
    def update(self, data: Dict[str, Any]) -> None
    def enable_checkpointing(self, checkpoint_dir: str, auto_save: bool = True)
    def save_checkpoint(self, name: Optional[str] = None) -> Path
    def load_checkpoint(self, name: str) -> None
```

### Agent Components

#### `@tool` Decorator

```python
@tool(
    name: Optional[str] = None,
    description: str,  # Required
    parameters: Optional[Dict[str, Any]] = None,
)
```

#### `ReActAgent` Class

```python
class ReActAgent:
    def __init__(self, llm: OpenAIClient, tool_registry: Optional[ToolRegistry] = None)
    async def run(self, question: str, streaming_callback: Optional[Callable] = None) -> str
    async def run_streaming(self, question: str, callback: Callable[[str, bool], None]) -> str
    def get_metrics(self) -> Dict[str, Any]
```

#### `OpenAIClient` Class

```python
class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo")
    async def chat(self, messages: List[Message], temperature: float = 0.7) -> LLMResponse
    async def stream(self, messages: List[Message], temperature: float = 0.7) -> AsyncGenerator[str, None]
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=LiteGraph --cov-report=html
```

## 📊 Observability

### Metrics

```python
# Graph metrics
metrics = graph.get_metrics()
print(f"Total executions: {metrics['total_executions']}")
print(f"Average execution time: {metrics['average_execution_time']:.3f}s")

# Agent metrics
agent_metrics = agent.get_metrics()
print(f"Agent executions: {agent_metrics['execution_count']}")
print(f"Tool success rate: {agent_metrics['tool_metrics']['success_rate']:.1%}")
```

### Checkpointing

```python
# Enable checkpointing
graph.config.enable_checkpointing = True
graph.config.checkpoint_dir = "./checkpoints"

# State will be automatically saved after each node
state = await graph.run()

# Manual checkpointing
state.save_checkpoint("my_checkpoint.json")
state.load_checkpoint("my_checkpoint.json")
```

### Hooks

```python
def before_hook(state, **kwargs):
    print(f"Before {kwargs['node_name']}: {state.data}")

def after_hook(state, **kwargs):
    print(f"After {kwargs['node_name']}: {state.data}")

@node(hooks={
    "before": [before_hook],
    "after": [after_hook]
})
async def my_node(state: State) -> dict:
    # Node logic here
    pass
```

## 🔌 Extending the Framework

### Custom LLM Adapters

```python
class CustomLLMClient:
    async def chat(self, messages: List[Message], temperature: float = 0.7) -> LLMResponse:
        # Your LLM implementation
        pass
    
    async def stream(self, messages: List[Message], temperature: float = 0.7):
        # Your streaming implementation
        pass

# Use with ReActAgent
agent = ReActAgent(CustomLLMClient())
```

### Custom Tools

```python
@tool(description="Your custom tool description")
async def custom_tool(param1: str, param2: int) -> str:
    # Your tool implementation
    return f"Processed {param1} with {param2}"
```

## 🚀 Production Features

- **Caching**: Node-level caching with TTL
- **Retries**: Configurable retry logic with exponential backoff
- **Timeouts**: Node and tool execution timeouts
- **Parallel Execution**: Independent nodes run concurrently
- **Loop Prevention**: Automatic cycle detection
- **State Persistence**: Optional checkpointing and recovery
- **Metrics Collection**: Built-in performance monitoring
- **Error Handling**: Graceful failure handling and recovery

## 📦 Dependencies

- `httpx` - HTTP client for LLM APIs
- `pydantic` - Data validation and settings
- `pytest` - Testing framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by modern AI orchestration platforms like LangGraph, with a focus on simplicity and performance. 
