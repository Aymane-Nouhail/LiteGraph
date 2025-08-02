"""
Advanced multi-agent workflow demo using Graph orchestration.

This demonstrates a more complex workflow similar to LangGraph's multi-agent systems,
where agents can hand off tasks to each other and coordinate their work.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from myagent import ReActAgent, tool, OpenAIClient, ToolRegistry, Graph, node, State

# Load environment variables
load_dotenv()


# ===== HANDOFF TOOL =====
@tool(description="Transfer the conversation to another agent")
def handoff_to_agent(agent_name: str, reason: str) -> str:
    """Transfer the conversation to another specialized agent."""
    return f"🔄 Transferring to {agent_name} agent. Reason: {reason}"


# ===== COORDINATOR AGENT =====
class CoordinatorAgent:
    """
    Main coordinator that manages the workflow and delegates tasks.
    """
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        self.tools.register(description="Transfer the conversation to another agent")(handoff_to_agent)
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def coordinate(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate the overall workflow."""
        
        system_prompt = f"""
You are the Coordinator Agent responsible for managing a multi-agent workflow.

Available specialized agents:
- text_processor: Handles text manipulation, hashing, and string operations
- math_analyzer: Handles number analysis, divisors, and prime factors  
- info_retriever: Handles weather information and data retrieval

Current query: "{query}"

Your task is to:
1. Analyze what type of work needs to be done
2. Decide which agent(s) should handle different parts
3. Use the handoff_to_agent tool to delegate work
4. Coordinate the overall response

If the query involves multiple types of work, you can hand off to multiple agents in sequence.
"""
        
        full_query = f"{system_prompt}\n\nUser: {query}"
        result = await self.agent.run(full_query)
        
        # Parse handoff instructions from the result
        handoffs = []
        if "handoff_to_agent" in result.lower():
            # Simple parsing - in a real system you'd use the tool execution results
            if "text_processor" in result.lower():
                handoffs.append("text_processor")
            if "math_analyzer" in result.lower():
                handoffs.append("math_analyzer")
            if "info_retriever" in result.lower():
                handoffs.append("info_retriever")
        
        state["coordinator_result"] = result
        state["handoffs"] = handoffs
        return state


# ===== SPECIALIZED AGENTS =====
@tool(description="Process and manipulate text in various ways")
def advanced_text_processing(text: str, operation: str) -> str:
    """Advanced text processing with multiple operations."""
    import hashlib
    
    operations = {
        "reverse": lambda t: t[::-1],
        "upper": lambda t: t.upper(),
        "lower": lambda t: t.lower(),
        "sha256": lambda t: hashlib.sha256(t.encode()).hexdigest(),
        "md5": lambda t: hashlib.md5(t.encode()).hexdigest(),
        "length": lambda t: str(len(t)),
        "words": lambda t: str(len(t.split())),
    }
    
    if operation in operations:
        result = operations[operation](text)
        return f"📝 {operation.upper()} of '{text}': {result}"
    else:
        available = ", ".join(operations.keys())
        return f"❌ Unknown operation '{operation}'. Available: {available}"


@tool(description="Perform advanced mathematical analysis on numbers")
def advanced_math_analysis(number: int, analysis_type: str) -> str:
    """Perform various mathematical analyses on a number."""
    import math
    
    analyses = {
        "divisors": lambda n: sorted({
            d for i in range(1, int(math.sqrt(n)) + 1)
            if n % i == 0 for d in (i, n // i)
        }),
        "prime_factors": lambda n: get_prime_factors(n),
        "is_prime": lambda n: is_prime(n),
        "is_perfect": lambda n: sum(d for d in range(1, n) if n % d == 0) == n,
        "digit_sum": lambda n: sum(int(d) for d in str(n)),
    }
    
    if analysis_type in analyses:
        result = analyses[analysis_type](number)
        return f"🔢 {analysis_type.upper()} of {number}: {result}"
    else:
        available = ", ".join(analyses.keys())
        return f"❌ Unknown analysis '{analysis_type}'. Available: {available}"


def get_prime_factors(n):
    """Get prime factors of a number."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


@tool(description="Retrieve various types of information")
async def advanced_info_retrieval(query_type: str, location: str = "default") -> str:
    """Retrieve information based on query type."""
    await asyncio.sleep(0.1)  # Simulate API call
    
    info_types = {
        "weather": f"🌤️ Weather in {location}: Sunny, 22°C, Light breeze",
        "humidity": f"💧 Humidity in {location}: 65%",
        "temperature": f"🌡️ Temperature in {location}: 22°C", 
        "forecast": f"📅 5-day forecast for {location}: Mostly sunny, 20-25°C",
        "time": f"🕐 Current time in {location}: 14:30 UTC",
    }
    
    if query_type in info_types:
        return info_types[query_type]
    else:
        available = ", ".join(info_types.keys())
        return f"❌ Unknown query type '{query_type}'. Available: {available}"


class TextProcessorAgent:
    """Specialized agent for text processing tasks."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        self.tools.register(description="Process and manipulate text in various ways")(advanced_text_processing)
        self.tools.register(description="Transfer the conversation to another agent")(handoff_to_agent)
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process(self, query: str, state: Dict[str, Any]) -> str:
        """Process text-related tasks."""
        system_prompt = """
You are a TextProcessor Agent specialized in text manipulation and analysis.

Available operations:
- reverse: Reverse the text
- upper/lower: Change case
- sha256/md5: Generate hashes
- length: Count characters
- words: Count words

Use the advanced_text_processing tool to handle text operations.
If you need to delegate to another agent, use handoff_to_agent.
"""
        
        full_query = f"{system_prompt}\n\nTask: {query}"
        result = await self.agent.run(full_query)
        state["text_processor_result"] = result
        return result


class MathAnalyzerAgent:
    """Specialized agent for mathematical analysis."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        self.tools.register(description="Perform advanced mathematical analysis on numbers")(advanced_math_analysis)
        self.tools.register(description="Transfer the conversation to another agent")(handoff_to_agent)
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process(self, query: str, state: Dict[str, Any]) -> str:
        """Process mathematical analysis tasks."""
        system_prompt = """
You are a MathAnalyzer Agent specialized in number theory and mathematical analysis.

Available analyses:
- divisors: Find all divisors
- prime_factors: Find prime factorization
- is_prime: Check if number is prime
- is_perfect: Check if number is perfect
- digit_sum: Sum of digits

Use the advanced_math_analysis tool to perform calculations.
If you need to delegate to another agent, use handoff_to_agent.
"""
        
        full_query = f"{system_prompt}\n\nTask: {query}"
        result = await self.agent.run(full_query)
        state["math_analyzer_result"] = result
        return result


class InfoRetrieverAgent:
    """Specialized agent for information retrieval."""
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.tools = ToolRegistry()
        self.tools.register(description="Retrieve various types of information")(advanced_info_retrieval)
        self.tools.register(description="Transfer the conversation to another agent")(handoff_to_agent)
        self.agent = ReActAgent(llm, tool_registry=self.tools)
    
    async def process(self, query: str, state: Dict[str, Any]) -> str:
        """Process information retrieval tasks."""
        system_prompt = """
You are an InfoRetriever Agent specialized in gathering various types of information.

Available information types:
- weather: Current weather conditions
- humidity: Humidity levels
- temperature: Temperature readings
- forecast: Weather forecasts
- time: Current time information

Use the advanced_info_retrieval tool to get information.
If you need to delegate to another agent, use handoff_to_agent.
"""
        
        full_query = f"{system_prompt}\n\nTask: {query}"
        result = await self.agent.run(full_query)
        state["info_retriever_result"] = result
        return result


# ===== WORKFLOW ORCHESTRATION =====
class MultiAgentWorkflow:
    """
    Orchestrates a multi-agent workflow with coordination and handoffs.
    """
    
    def __init__(self, llm: OpenAIClient):
        self.llm = llm
        self.coordinator = CoordinatorAgent(llm)
        self.agents = {
            "text_processor": TextProcessorAgent(llm),
            "math_analyzer": MathAnalyzerAgent(llm),
            "info_retriever": InfoRetrieverAgent(llm),
        }
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute a multi-agent workflow."""
        
        print(f"🎯 Starting workflow for: {query}")
        
        # Initialize workflow state
        state = {
            "original_query": query,
            "steps": [],
            "results": {},
        }
        
        # Step 1: Coordination
        print("1️⃣ Coordinator analyzing query...")
        state = await self.coordinator.coordinate(query, state)
        
        # Step 2: Execute handoffs
        handoffs = state.get("handoffs", [])
        if handoffs:
            print(f"2️⃣ Executing handoffs to: {', '.join(handoffs)}")
            
            for agent_name in handoffs:
                if agent_name in self.agents:
                    print(f"   🔄 Processing with {agent_name}...")
                    agent = self.agents[agent_name]
                    result = await agent.process(query, state)
                    state["results"][agent_name] = result
                    state["steps"].append(f"Processed by {agent_name}")
        else:
            print("2️⃣ No handoffs required, coordinator handled directly")
        
        # Step 3: Summarize results
        print("3️⃣ Workflow completed")
        return state


async def demo_complex_workflows():
    """Demonstrate complex multi-agent workflows."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize system
    llm = OpenAIClient(api_key=api_key, model="gpt-3.5-turbo")
    workflow = MultiAgentWorkflow(llm)
    
    # Test complex queries requiring coordination
    complex_queries = [
        "Analyze the number 60 and also reverse the text 'hello'",
        "Get the weather in Tokyo and find the prime factors of 42",
        "Convert 'workflow' to uppercase and check if 17 is prime",
        "What's the humidity in Paris and compute the MD5 of 'test'?",
    ]
    
    print("🚀 Demonstrating complex multi-agent workflows")
    print("=" * 60)
    
    for i, query in enumerate(complex_queries, 1):
        print(f"\n🔍 Complex Query {i}/{len(complex_queries)}")
        print("-" * 40)
        
        try:
            result_state = await workflow.execute(query)
            
            print("\n📊 Workflow Summary:")
            print(f"   Original Query: {result_state['original_query']}")
            print(f"   Steps Executed: {len(result_state['steps'])}")
            
            for agent_name, result in result_state['results'].items():
                print(f"   {agent_name}: {result}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error in workflow: {e}")
            print("=" * 60)


async def main():
    """Main demo function."""
    await demo_complex_workflows()


if __name__ == "__main__":
    asyncio.run(main())
