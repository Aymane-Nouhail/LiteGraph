"""
ReAct agent demo with tool integration.

Demonstrates:
- Tool registration with @tool decorator
- ReAct agent with streaming
- Tool execution and error handling
- Multi-step reasoning
"""

import asyncio
import json
from typing import Dict, Any
from LiteGraph import ReActAgent, tool, OpenAIClient


# Example tools
@tool(description="Get the current weather for a location")
async def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Simulate API call
    await asyncio.sleep(0.5)
    return f"Weather in {location}: Sunny, 25°C"


@tool(description="Calculate mathematical expressions")
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Safe evaluation of basic math expressions
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(description="Search for information on a topic")
async def search_info(topic: str) -> str:
    """Search for information on a topic."""
    # Simulate search
    await asyncio.sleep(0.3)
    return f"Search results for '{topic}': Found relevant information about {topic}"


@tool(description="Convert temperature between Celsius and Fahrenheit")
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between units."""
    try:
        if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
            result = (value * 9/5) + 32
            return f"{value}°C = {result:.1f}°F"
        elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
            result = (value - 32) * 5/9
            return f"{value}°F = {result:.1f}°C"
        else:
            return f"Error: Unsupported conversion from {from_unit} to {to_unit}"
    except Exception as e:
        return f"Error: {str(e)}"


def streaming_callback(chunk: str, is_complete: bool):
    """Callback for streaming responses."""
    if is_complete:
        print("\n✅ Agent completed!")
    else:
        print(chunk, end="", flush=True)


async def main():
    """Run the ReAct agent demo."""
    print("🧠 LiteGraph - ReAct Agent Demo")
    print("=" * 50)
    
    # Initialize LLM client (you'll need to set OPENAI_API_KEY)
    try:
        llm = OpenAIClient()
    except ValueError as e:
        print(f"❌ {e}")
        print("Please set the OPENAI_API_KEY environment variable to run this demo.")
        return
    
    # Create ReAct agent
    agent = ReActAgent(llm)
    
    # Example questions
    questions = [
        "What's 15 * 23?",
        "What's the weather like in Paris?",
        "Convert 25 degrees Celsius to Fahrenheit",
        "Search for information about artificial intelligence and then tell me what you found"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Run agent with streaming
            print("🤖 Agent thinking...")
            answer = await agent.run_streaming(question, streaming_callback)
            
            print(f"\n💡 Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
    
    # Display agent metrics
    print("\n📊 Agent Metrics:")
    metrics = agent.get_metrics()
    print(f"  Total executions: {metrics.get('execution_count', 0)}")
    print(f"  Average execution time: {metrics.get('average_execution_time', 0):.3f}s")
    
    # Display tool metrics
    tool_metrics = metrics.get('tool_metrics', {})
    print(f"  Tool executions: {tool_metrics.get('total_executions', 0)}")
    print(f"  Tool success rate: {tool_metrics.get('success_rate', 0):.1%}")
    print(f"  Average tool execution time: {tool_metrics.get('average_execution_time', 0):.3f}s")
    
    # Display tool usage
    print("\n🔧 Tool Usage:")
    tool_usage = tool_metrics.get('tool_usage', {})
    for tool_name, count in tool_usage.items():
        print(f"  {tool_name}: {count} calls")


async def demo_without_openai():
    """Demo without requiring OpenAI API key."""
    print("🧠 LiteGraph - ReAct Agent Demo (Simulated)")
    print("=" * 50)
    
    # Create a mock LLM client
    class MockLLMClient:
        async def chat(self, messages, temperature=0.7, max_tokens=None):
            from LiteGraph.llms.openai_client import LLMResponse
            # Simulate a ReAct response
            response = """Thought: I need to calculate 15 * 23
Action: calculate
Action Input: {"expression": "15 * 23"}
Observation: Result: 345
Final Answer: 15 * 23 = 345"""
            return LLMResponse(content=response)
        
        async def stream(self, messages, temperature=0.7, max_tokens=None):
            response = """Thought: I need to calculate 15 * 23
Action: calculate
Action Input: {"expression": "15 * 23"}
Observation: Result: 345
Final Answer: 15 * 23 = 345"""
            for char in response:
                yield char
                await asyncio.sleep(0.01)
    
    # Create ReAct agent with mock LLM
    agent = ReActAgent(MockLLMClient())
    
    question = "What's 15 * 23?"
    print(f"🤔 Question: {question}")
    print("-" * 40)
    
    try:
        print("🤖 Agent thinking...")
        answer = await agent.run_streaming(question, streaming_callback)
        print(f"\n💡 Answer: {answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Display metrics
    print("\n📊 Agent Metrics:")
    metrics = agent.get_metrics()
    print(f"  Total executions: {metrics.get('execution_count', 0)}")
    print(f"  Average execution time: {metrics.get('average_execution_time', 0):.3f}s")


if __name__ == "__main__":
    # Try to run with real OpenAI client, fallback to mock
    try:
        asyncio.run(main())
    except Exception:
        print("⚠️  Falling back to simulated demo...")
        asyncio.run(demo_without_openai()) 