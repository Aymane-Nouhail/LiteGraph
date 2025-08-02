"""
Test execution of the agent framework with tools analogous to LangGraph example.

This demonstrates:
- Text Editor Agent (reverse text, SHA256, MD5, string to number)
- Divisor Agent (list divisors, prime factors)
- Weather Agent (get weather, humidity, current city)
- Multi-agent workflow coordination
"""

import asyncio
import os
import hashlib
import math
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from LiteGraph import ReActAgent, tool, OpenAIClient, ToolRegistry, Graph, node, State

# Load environment variables
load_dotenv()


# ===== TEXT EDITOR TOOLS =====
@tool(description="Reverses the input text")
def reverse_text(text: str) -> str:
    """Reverses the input text."""
    reversed_str = text[::-1]
    return f"🔄 Reversed '{text}' → '{reversed_str}'"


@tool(description="Computes the SHA-256 hash of the input text")
def compute_sha256(text: str) -> str:
    """Computes the SHA-256 hash of the input text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"🔐 SHA-256('{text}') → {digest}"


@tool(description="Computes the MD5 hash of the input text")
def compute_md5(text: str) -> str:
    """Computes the MD5 hash of the input text."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"🗝️ MD5('{text}') → {digest}"


@tool(description="Converts a string to a number using a simple hash function")
def string_to_number(text: str) -> str:
    """Converts a string to a number using a simple hash function."""
    # Using a simple hash function to convert string to number
    number = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10**10)
    return f"🔢 String to number '{text}' → {number}"


# ===== DIVISOR TOOLS =====
@tool(description="Lists all divisors of the given number")
def list_divisors(number: int) -> str:
    """Lists all divisors of the given number."""
    divs = sorted({
        d
        for i in range(1, int(math.sqrt(number)) + 1)
        if number % i == 0
        for d in (i, number // i)
    })
    return f"📜 Divisors of {number}: {divs}"


@tool(description="Computes the prime factors of the given number")
def prime_factors(number: int) -> str:
    """Computes the prime factors of the given number."""
    n, factors = number, []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return f"⚙️ Prime factors of {number}: {factors}"


# ===== WEATHER TOOLS =====
@tool(description="Fetches the current weather for the given city")
async def get_weather(city: str) -> str:
    """Fetches the current weather for the given city."""
    # Simulate API call delay
    await asyncio.sleep(0.2)
    # This is a placeholder implementation
    return f"🌤️ Current weather in {city}: Sunny, 25°C"


@tool(description="Fetches the current humidity for the given city")
async def get_humidity(city: str) -> str:
    """Fetches the current humidity for the given city."""
    # Simulate API call delay
    await asyncio.sleep(0.1)
    # This is a placeholder implementation
    return f"💧 Current humidity in {city}: 60%"


@tool(description="Returns the current city for weather queries")
def get_current_city() -> str:
    """Returns the current city for weather queries."""
    # This is a placeholder implementation
    return "🏙️ Current city: Paris"


# ===== AGENT STATE AND CONFIGURATION =====
class AgentState(State):
    """Custom state for multi-agent coordination."""
    
    def __init__(self):
        super().__init__()
        self.data.update({
            "messages": [],
            "remaining_steps": 10,
            "active_agent": "router",
            "agent_results": {},
            "task_type": None
        })


class TextEditorAgent:
    """Text Editor Agent that handles text manipulation tasks."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        # Register text tools with this agent's tool registry
        self.tools.register(description="Reverses the input text")(reverse_text)
        self.tools.register(description="Computes the SHA-256 hash of the input text")(compute_sha256)
        self.tools.register(description="Computes the MD5 hash of the input text")(compute_md5)
        self.tools.register(description="Converts a string to a number using a simple hash function")(string_to_number)
        
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process(self, state: AgentState) -> str:
        """Process text-related tasks."""
        system_prompt = """You are a TextEditor. Available tools:
- reverse_text: Reverses the input text
- compute_sha256: Computes the SHA-256 hash of the input text
- compute_md5: Computes the MD5 hash of the input text
- string_to_number: Converts a string to a number using a hash function

Choose the appropriate tool based on the user message. Be helpful and use the tools effectively."""
        
        # Get the last user message
        messages = state.get("messages", [])
        if messages:
            user_query = messages[-1]
            full_query = f"{system_prompt}\n\nUser: {user_query}"
            result = await self.agent.run(full_query)
            return result
        return "No query provided"


class DivisorAgent:
    """Divisor Agent that handles number analysis tasks."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        # Register divisor tools with this agent's tool registry
        self.tools.register(description="Lists all divisors of the given number")(list_divisors)
        self.tools.register(description="Computes the prime factors of the given number")(prime_factors)
        
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process(self, state: AgentState) -> str:
        """Process number analysis tasks."""
        system_prompt = """You are a DivisorAgent. Available tools:
- list_divisors: Lists all divisors of a given number
- prime_factors: Computes the prime factors of a given number

Select and call the proper tool based on the user message. Be thorough in your analysis."""
        
        # Get the last user message
        messages = state.get("messages", [])
        if messages:
            user_query = messages[-1]
            full_query = f"{system_prompt}\n\nUser: {user_query}"
            result = await self.agent.run(full_query)
            return result
        return "No query provided"


class WeatherAgent:
    """Weather Agent that handles weather-related queries."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        # Register weather tools with this agent's tool registry
        self.tools.register(description="Fetches the current weather for the given city")(get_weather)
        self.tools.register(description="Fetches the current humidity for the given city")(get_humidity)
        self.tools.register(description="Returns the current city for weather queries")(get_current_city)
        
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process(self, state: AgentState) -> str:
        """Process weather-related tasks."""
        system_prompt = """You are a WeatherAgent. Available tools:
- get_weather: Fetches current weather for a given city
- get_humidity: Fetches current humidity for a given city  
- get_current_city: Returns the current city for weather queries

Use these tools to provide accurate weather information based on user queries."""
        
        # Get the last user message
        messages = state.get("messages", [])
        if messages:
            user_query = messages[-1]
            full_query = f"{system_prompt}\n\nUser: {user_query}"
            result = await self.agent.run(full_query)
            return result
        return "No query provided"


class RouterAgent:
    """Router agent that decides which specialized agent to use."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
    
    async def route(self, query: str) -> str:
        """Determine which agent should handle the query."""
        routing_prompt = f"""
Analyze this user query and determine which agent should handle it:

Query: "{query}"

Available agents:
- text_editor: For text manipulation (reversing, hashing, string conversion)
- divisor: For number analysis (finding divisors, prime factors)
- weather: For weather information (current weather, humidity, location)

Respond with ONLY one of: text_editor, divisor, weather
"""
        
        # Use a simple classification approach
        query_lower = query.lower()
        if any(word in query_lower for word in ["reverse", "hash", "sha256", "md5", "string", "text"]):
            return "text_editor"
        elif any(word in query_lower for word in ["divisor", "factor", "prime", "number", "math"]):
            return "divisor"
        elif any(word in query_lower for word in ["weather", "temperature", "humidity", "city", "forecast"]):
            return "weather"
        else:
            # Default to text editor for ambiguous cases
            return "text_editor"


async def create_multi_agent_system(llm: OpenAIClient) -> Dict[str, Any]:
    """Create a multi-agent system with specialized agents."""
    
    # Create specialized agents
    text_editor = TextEditorAgent(llm)
    divisor = DivisorAgent(llm)
    weather = WeatherAgent(llm)
    router = RouterAgent(llm)
    
    return {
        "text_editor": text_editor,
        "divisor": divisor,
        "weather": weather,
        "router": router
    }


async def run_multi_agent_workflow(agents: Dict[str, Any], query: str) -> str:
    """Run a multi-agent workflow to process a query."""
    
    print(f"🤖 Processing query: {query}")
    
    # Route the query to the appropriate agent
    router = agents["router"]
    agent_type = await router.route(query)
    print(f"📍 Routing to: {agent_type} agent")
    
    # Create state and add the query
    state = AgentState()
    state.data["messages"].append(query)
    state.data["active_agent"] = agent_type
    
    # Process with the selected agent
    selected_agent = agents[agent_type]
    result = await selected_agent.process(state)
    
    print(f"✅ Result from {agent_type} agent:")
    print(result)
    print("-" * 50)
    
    return result


async def main():
    """Main test execution function."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        print("You can create a .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize OpenAI client
    llm = OpenAIClient(api_key=api_key, model="gpt-3.5-turbo")
    
    # Create multi-agent system
    print("🚀 Creating multi-agent system...")
    agents = await create_multi_agent_system(llm)
    
    # Test queries for different agents
    test_queries = [
        # Text Editor Agent tests
        "Please reverse the text 'hello world'",
        "Compute the SHA256 hash of 'LiteGraph'", 
        "Convert the string 'framework' to a number",
        
        # Divisor Agent tests
        "Find all divisors of 24",
        "What are the prime factors of 60?",
        "Give me the divisors and prime factors of 100",
        
        # Weather Agent tests
        "What's the weather in Paris?",
        "Get the humidity in London",
        "What's the current city and its weather?",
    ]
    
    print("🧪 Running test scenarios...")
    print("=" * 60)
    
    # Execute test queries
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}/{len(test_queries)}")
        try:
            await run_multi_agent_workflow(agents, query)
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("-" * 50)
    
    print("\n🎉 Test execution completed!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
