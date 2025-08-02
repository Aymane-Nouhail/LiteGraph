"""
Tests for ReAct parser and agent components.
"""

import pytest
from myagent.agent.parser import ReActParser, ReActStep


@pytest.fixture
def parser():
    """Create a ReAct parser instance."""
    return ReActParser()


def test_parse_simple_react(parser):
    """Test parsing a simple ReAct response."""
    text = """Thought: I need to calculate 15 * 23
Action: calculate
Action Input: {"expression": "15 * 23"}
Observation: Result: 345
Final Answer: 15 * 23 = 345"""
    
    steps = parser.parse(text)
    
    assert len(steps) == 1
    step = steps[0]
    assert step.thought == "I need to calculate 15 * 23"
    assert step.action == "calculate"
    assert step.action_input == {"expression": "15 * 23"}
    assert step.observation == "Result: 345"
    assert step.final_answer == "15 * 23 = 345"


def test_parse_multiple_steps(parser):
    """Test parsing multiple ReAct steps."""
    text = """Thought: I need to get the weather first
Action: get_weather
Action Input: {"location": "Paris"}
Observation: Weather in Paris: Sunny, 25°C

Thought: Now I can provide the answer
Final Answer: The weather in Paris is sunny with a temperature of 25°C"""
    
    steps = parser.parse(text)
    
    assert len(steps) == 2
    
    # First step
    step1 = steps[0]
    assert step1.thought == "I need to get the weather first"
    assert step1.action == "get_weather"
    assert step1.action_input == {"location": "Paris"}
    assert step1.observation == "Weather in Paris: Sunny, 25°C"
    assert step1.final_answer is None
    
    # Second step
    step2 = steps[1]
    assert step2.thought == "Now I can provide the answer"
    assert step2.action is None
    assert step2.action_input is None
    assert step2.observation is None
    assert step2.final_answer == "The weather in Paris is sunny with a temperature of 25°C"


def test_parse_streaming(parser):
    """Test parsing streaming responses."""
    # Incomplete response
    text = "Thought: I need to calculate"
    steps, is_complete = parser.parse_streaming(text)
    
    assert len(steps) == 1
    assert steps[0].thought == "I need to calculate"
    assert not is_complete
    
    # Complete response
    text = """Thought: I need to calculate 15 * 23
Action: calculate
Action Input: {"expression": "15 * 23"}
Observation: Result: 345
Final Answer: 15 * 23 = 345"""
    
    steps, is_complete = parser.parse_streaming(text)
    
    assert len(steps) == 1
    assert is_complete


def test_extract_final_answer(parser):
    """Test extracting final answer from response."""
    text = """Thought: I need to calculate 15 * 23
Action: calculate
Action Input: {"expression": "15 * 23"}
Observation: Result: 345
Final Answer: 15 * 23 = 345"""
    
    answer = parser.extract_final_answer(text)
    assert answer == "15 * 23 = 345"
    
    # No final answer
    text = """Thought: I need to calculate 15 * 23
Action: calculate
Action Input: {"expression": "15 * 23"}
Observation: Result: 345"""
    
    answer = parser.extract_final_answer(text)
    assert answer is None


def test_extract_tool_calls(parser):
    """Test extracting tool calls from response."""
    text = """Thought: I need to get weather and calculate
Action: get_weather
Action Input: {"location": "Paris"}
Observation: Weather in Paris: Sunny, 25°C

Thought: Now calculate temperature conversion
Action: convert_temperature
Action Input: {"value": 25, "from_unit": "celsius", "to_unit": "fahrenheit"}
Observation: 25°C = 77.0°F

Final Answer: The weather in Paris is sunny at 77°F"""
    
    tool_calls = parser.extract_tool_calls(text)
    
    assert len(tool_calls) == 2
    assert tool_calls[0] == ("get_weather", {"location": "Paris"})
    assert tool_calls[1] == ("convert_temperature", {"value": 25, "from_unit": "celsius", "to_unit": "fahrenheit"})


def test_format_tools_prompt(parser):
    """Test formatting tools for prompt."""
    tools = [
        {"name": "calculate", "description": "Calculate mathematical expressions"},
        {"name": "get_weather", "description": "Get weather for a location"}
    ]
    
    formatted = parser.format_tools_prompt(tools)
    
    assert "Available tools:" in formatted
    assert "- **calculate**: Calculate mathematical expressions" in formatted
    assert "- **get_weather**: Get weather for a location" in formatted


def test_create_react_prompt(parser):
    """Test creating ReAct prompt."""
    tools = [
        {"name": "calculate", "description": "Calculate mathematical expressions"}
    ]
    
    prompt = parser.create_react_prompt("What's 15 * 23?", tools)
    
    assert "You are a helpful AI assistant" in prompt
    assert "Thought: I need to think about this step by step" in prompt
    assert "Action: tool_name" in prompt
    assert "Action Input: {\"arg1\": \"value1\", \"arg2\": \"value2\"}" in prompt
    assert "Observation: tool_result" in prompt
    assert "Final Answer: final_response" in prompt
    assert "Question: What's 15 * 23?" in prompt
    assert "- **calculate**: Calculate mathematical expressions" in prompt


def test_create_react_prompt_with_examples(parser):
    """Test creating ReAct prompt with examples."""
    tools = [
        {"name": "calculate", "description": "Calculate mathematical expressions"}
    ]
    
    examples = [
        "Thought: I need to calculate 10 + 5\nAction: calculate\nAction Input: {\"expression\": \"10 + 5\"}\nObservation: Result: 15\nFinal Answer: 10 + 5 = 15"
    ]
    
    prompt = parser.create_react_prompt("What's 15 * 23?", tools, examples)
    
    assert "Examples:" in prompt
    assert "Thought: I need to calculate 10 + 5" in prompt


def test_parse_malformed_json(parser):
    """Test parsing with malformed JSON in action input."""
    text = """Thought: I need to search
Action: search
Action Input: {"query": "test query"}
Observation: Search results
Final Answer: Found results"""
    
    steps = parser.parse(text)
    
    assert len(steps) == 1
    step = steps[0]
    assert step.action_input == {"query": "test query"}
    
    # Test with invalid JSON
    text = """Thought: I need to search
Action: search
Action Input: invalid json
Observation: Search results
Final Answer: Found results"""
    
    steps = parser.parse(text)
    
    assert len(steps) == 1
    step = steps[0]
    assert step.action_input == {"input": "invalid json"}


def test_parse_empty_response(parser):
    """Test parsing empty response."""
    steps = parser.parse("")
    assert len(steps) == 0
    
    steps, is_complete = parser.parse_streaming("")
    assert len(steps) == 0
    assert not is_complete


def test_parse_partial_thought(parser):
    """Test parsing partial thought."""
    text = "Thought: This is a partial thought"
    steps = parser.parse(text)
    
    assert len(steps) == 1
    assert steps[0].thought == "This is a partial thought"
    assert steps[0].action is None
    assert steps[0].final_answer is None


def test_parse_with_extra_whitespace(parser):
    """Test parsing with extra whitespace."""
    text = """
    Thought: I need to calculate 15 * 23
    
    Action: calculate
    
    Action Input: {"expression": "15 * 23"}
    
    Observation: Result: 345
    
    Final Answer: 15 * 23 = 345
    """
    
    steps = parser.parse(text)
    
    assert len(steps) == 1
    step = steps[0]
    assert step.thought == "I need to calculate 15 * 23"
    assert step.action == "calculate"
    assert step.action_input == {"expression": "15 * 23"}
    assert step.observation == "Result: 345"
    assert step.final_answer == "15 * 23 = 345" 