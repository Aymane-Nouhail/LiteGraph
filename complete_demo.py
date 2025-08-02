"""
Complete demonstration of the agent framework with tools analogous to LangGraph.

This shows:
1. Individual agent demonstrations
2. Tool execution examples  
3. Multi-step workflows
4. Agent coordination patterns
"""

import asyncio
import os
import hashlib
import math
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from LiteGraph import ReActAgent, tool, OpenAIClient, ToolRegistry

# Load environment variables
load_dotenv()


# ===== TEXT PROCESSING TOOLS (analogous to LangGraph text tools) =====
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
    number = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10**10)
    return f"🔢 String to number '{text}' → {number}"


# ===== MATH ANALYSIS TOOLS (analogous to LangGraph divisor tools) =====
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


# ===== WEATHER TOOLS (analogous to LangGraph weather tools) =====
@tool(description="Fetches the current weather for the given city")
async def get_weather(city: str) -> str:
    """Fetches the current weather for the given city."""
    await asyncio.sleep(0.1)  # Simulate API call
    return f"🌤️ Current weather in {city}: Sunny, 25°C"


@tool(description="Fetches the current humidity for the given city")
async def get_humidity(city: str) -> str:
    """Fetches the current humidity for the given city."""
    await asyncio.sleep(0.1)  # Simulate API call
    return f"💧 Current humidity in {city}: 60%"


@tool(description="Returns the current city for weather queries")
def get_current_city() -> str:
    """Returns the current city for weather queries."""
    return "🏙️ Current city: Paris"


# ===== SPECIALIZED AGENTS =====
class TextEditorAgent:
    """Text Editor Agent - handles text manipulation tasks."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        
        # Register text processing tools
        self.tools.register(description="Reverses the input text")(reverse_text)
        self.tools.register(description="Computes the SHA-256 hash of the input text")(compute_sha256)
        self.tools.register(description="Computes the MD5 hash of the input text")(compute_md5)
        self.tools.register(description="Converts a string to a number using a simple hash function")(string_to_number)
        
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process_text(self, text: str, operation: str) -> str:
        """Process text with specified operation."""
        prompt = f"""
You are a TextEditor agent. Available tools:
- reverse_text: Reverses the input text
- compute_sha256: Computes the SHA-256 hash of the input text  
- compute_md5: Computes the MD5 hash of the input text
- string_to_number: Converts a string to a number using a hash function

Task: {operation} the text "{text}"

Use the appropriate tool to complete this task.
"""
        return await self.agent.run(prompt)


class DivisorAgent:
    """Divisor Agent - handles number analysis tasks."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        
        # Register math analysis tools
        self.tools.register(description="Lists all divisors of the given number")(list_divisors)
        self.tools.register(description="Computes the prime factors of the given number")(prime_factors)
        
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def analyze_number(self, number: int, analysis_type: str) -> str:
        """Analyze number with specified analysis type."""
        prompt = f"""
You are a DivisorAgent. Available tools:
- list_divisors: Lists all divisors of a given number
- prime_factors: Computes the prime factors of a given number

Task: {analysis_type} for the number {number}

Use the appropriate tool to complete this analysis.
"""
        return await self.agent.run(prompt)


class WeatherAgent:
    """Weather Agent - handles weather information queries."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        
        # Register weather tools
        self.tools.register(description="Fetches the current weather for the given city")(get_weather)
        self.tools.register(description="Fetches the current humidity for the given city")(get_humidity)
        self.tools.register(description="Returns the current city for weather queries")(get_current_city)
        
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def get_weather_info(self, city: str, info_type: str) -> str:
        """Get weather information for specified city and type."""
        prompt = f"""
You are a WeatherAgent. Available tools:
- get_weather: Fetches current weather for a given city
- get_humidity: Fetches current humidity for a given city
- get_current_city: Returns the current city for weather queries

Task: Get {info_type} information for {city}

Use the appropriate tool to complete this request.
"""
        return await self.agent.run(prompt)


# ===== MULTI-AGENT COORDINATOR =====
class MultiAgentSystem:
    """Coordinates multiple specialized agents."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.text_editor = TextEditorAgent(llm)
        self.divisor_agent = DivisorAgent(llm)
        self.weather_agent = WeatherAgent(llm)
    
    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process a request that might require multiple agents."""
        print(f"🎯 Processing: {request}")
        
        results = {"request": request, "agent_results": []}
        
        # Simple keyword-based routing (in production, use an LLM for this)
        request_lower = request.lower()
        
        # Check for text operations
        if any(word in request_lower for word in ["reverse", "hash", "sha256", "md5", "string"]):
            print("   🔄 Routing to TextEditor agent...")
            
            # Extract text and operation
            if "reverse" in request_lower:
                if "hello" in request_lower:
                    result = await self.text_editor.process_text("hello", "reverse")
                elif "world" in request_lower:
                    result = await self.text_editor.process_text("world", "reverse")
                else:
                    result = await self.text_editor.process_text("example", "reverse")
            elif "sha256" in request_lower:
                text = "LiteGraph" if "LiteGraph" in request_lower else "example"
                result = await self.text_editor.process_text(text, "compute SHA256 hash")
            elif "md5" in request_lower:
                text = "test" if "test" in request_lower else "example"
                result = await self.text_editor.process_text(text, "compute MD5 hash")
            else:
                result = await self.text_editor.process_text("framework", "convert to number")
            
            results["agent_results"].append({"agent": "TextEditor", "result": result})
        
        # Check for math operations
        if any(word in request_lower for word in ["divisor", "factor", "prime", "analyze"]):
            print("   🔢 Routing to Divisor agent...")
            
            # Extract number and operation
            number = 60  # default
            if "24" in request:
                number = 24
            elif "42" in request:
                number = 42
            elif "100" in request:
                number = 100
            
            if "divisor" in request_lower:
                result = await self.divisor_agent.analyze_number(number, "find all divisors")
            else:
                result = await self.divisor_agent.analyze_number(number, "find prime factors")
            
            results["agent_results"].append({"agent": "Divisor", "result": result})
        
        # Check for weather operations
        if any(word in request_lower for word in ["weather", "humidity", "temperature", "tokyo", "paris", "london"]):
            print("   🌤️ Routing to Weather agent...")
            
            city = "Paris"  # default
            if "tokyo" in request_lower:
                city = "Tokyo"
            elif "london" in request_lower:
                city = "London"
            
            if "humidity" in request_lower:
                result = await self.weather_agent.get_weather_info(city, "humidity")
            else:
                result = await self.weather_agent.get_weather_info(city, "weather")
            
            results["agent_results"].append({"agent": "Weather", "result": result})
        
        return results


async def demonstrate_individual_agents():
    """Demonstrate each agent individually."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        return
    
    llm = OpenAIClient(api_key=api_key, model="gpt-3.5-turbo")
    
    print("🧪 INDIVIDUAL AGENT DEMONSTRATIONS")
    print("=" * 60)
    
    # Text Editor Agent Demo
    print("\n1️⃣ TEXT EDITOR AGENT")
    print("-" * 30)
    text_agent = TextEditorAgent(llm)
    
    text_demos = [
        ("hello world", "reverse"),
        ("LiteGraph", "compute SHA256 hash"),
        ("framework", "convert to number"),
    ]
    
    for text, operation in text_demos:
        print(f"📝 {operation} '{text}':")
        result = await text_agent.process_text(text, operation)
        print(f"   ✅ {result}")
    
    # Divisor Agent Demo
    print("\n2️⃣ DIVISOR AGENT")
    print("-" * 30)
    divisor_agent = DivisorAgent(llm)
    
    number_demos = [
        (24, "find all divisors"),
        (60, "find prime factors"),
        (100, "find all divisors and prime factors"),
    ]
    
    for number, analysis in number_demos:
        print(f"🔢 {analysis} of {number}:")
        result = await divisor_agent.analyze_number(number, analysis)
        print(f"   ✅ {result}")
    
    # Weather Agent Demo
    print("\n3️⃣ WEATHER AGENT") 
    print("-" * 30)
    weather_agent = WeatherAgent(llm)
    
    weather_demos = [
        ("Paris", "weather"),
        ("Tokyo", "humidity"),
        ("London", "weather"),
    ]
    
    for city, info_type in weather_demos:
        print(f"🌤️ Get {info_type} for {city}:")
        result = await weather_agent.get_weather_info(city, info_type)
        print(f"   ✅ {result}")


async def demonstrate_multi_agent_coordination():
    """Demonstrate multi-agent coordination."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        return
    
    llm = OpenAIClient(api_key=api_key, model="gpt-3.5-turbo")
    
    print("\n\n🤝 MULTI-AGENT COORDINATION")
    print("=" * 60)
    
    system = MultiAgentSystem(llm)
    
    # Complex requests requiring multiple agents
    complex_requests = [
        "Reverse the text 'hello' and find divisors of 24",
        "Get weather in Tokyo and compute SHA256 of 'LiteGraph'",
        "Find prime factors of 42 and get humidity in Paris",
        "Convert 'framework' to number and check weather in London",
    ]
    
    for i, request in enumerate(complex_requests, 1):
        print(f"\n🔍 Complex Request {i}: {request}")
        print("-" * 50)
        
        results = await system.process_request(request)
        
        print("📊 Results:")
        for agent_result in results["agent_results"]:
            agent_name = agent_result["agent"]
            result = agent_result["result"]
            print(f"   {agent_name}: {result}")


async def main():
    """Main demonstration function."""
    print("🚀 AGENT FRAMEWORK DEMONSTRATION")
    print("Analogous to LangGraph multi-agent systems")
    print("=" * 60)
    
    # Individual agent demonstrations
    await demonstrate_individual_agents()
    
    # Multi-agent coordination
    await demonstrate_multi_agent_coordination()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("\n💡 This framework provides:")
    print("   ✅ Individual specialized agents")
    print("   ✅ Tool registration and execution")
    print("   ✅ Multi-agent coordination")
    print("   ✅ Streaming support")
    print("   ✅ Error handling and retries")
    print("   ✅ Extensible architecture")


if __name__ == "__main__":
    asyncio.run(main())
