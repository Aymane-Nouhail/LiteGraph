"""
ReAct parser for parsing agent responses into structured components.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ReActStep(BaseModel):
    """A single ReAct step (Thought/Action/Observation)."""
    
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None


class ReActParser:
    """
    Parser for ReAct (Reasoning and Acting) agent responses.
    
    Parses responses in the format:
    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <tool_args>
    Observation: <tool_result>
    ... (repeat)
    Final Answer: <final_response>
    """
    
    def __init__(self):
        # Regex patterns for parsing ReAct format
        self.thought_pattern = re.compile(r"Thought:\s*(.+)", re.DOTALL)
        self.action_pattern = re.compile(r"Action:\s*(\w+)", re.DOTALL)
        self.action_input_pattern = re.compile(r"Action Input:\s*(.+)", re.DOTALL)
        self.observation_pattern = re.compile(r"Observation:\s*(.+)", re.DOTALL)
        self.final_answer_pattern = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
    
    def parse(self, text: str) -> List[ReActStep]:
        """
        Parse ReAct response into structured steps.
        
        Args:
            text: Raw response text from agent
        
        Returns:
            List of ReActStep objects
        """
        steps = []
        current_step = ReActStep()
        
        # Split by lines and process
        lines = text.strip().split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for Thought
            thought_match = self.thought_pattern.match(line)
            if thought_match:
                if current_step.thought or current_step.action:
                    # Start new step
                    if current_step.thought or current_step.action or current_step.observation:
                        steps.append(current_step)
                    current_step = ReActStep()
                current_step.thought = thought_match.group(1).strip()
                i += 1
                continue
            
            # Check for Action
            action_match = self.action_pattern.match(line)
            if action_match:
                current_step.action = action_match.group(1).strip()
                i += 1
                continue
            
            # Check for Action Input
            action_input_match = self.action_input_pattern.match(line)
            if action_input_match:
                action_input_text = action_input_match.group(1).strip()
                try:
                    # Try to parse as JSON
                    import json
                    current_step.action_input = json.loads(action_input_text)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as string
                    current_step.action_input = {"input": action_input_text}
                i += 1
                continue
            
            # Check for Observation
            observation_match = self.observation_pattern.match(line)
            if observation_match:
                current_step.observation = observation_match.group(1).strip()
                i += 1
                continue
            
            # Check for Final Answer
            final_answer_match = self.final_answer_pattern.match(line)
            if final_answer_match:
                current_step.final_answer = final_answer_match.group(1).strip()
                steps.append(current_step)
                break
            
            # If no pattern matches, continue to next line
            i += 1
        
        # Add the last step if it has content and wasn't already added
        if (current_step.thought or current_step.action or current_step.observation or current_step.final_answer) and current_step not in steps:
            steps.append(current_step)
        
        return steps
    
    def parse_streaming(self, text: str) -> Tuple[List[ReActStep], bool]:
        """
        Parse streaming ReAct response, returning completed steps and whether parsing is complete.
        
        Args:
            text: Partial or complete response text
        
        Returns:
            Tuple of (completed_steps, is_complete)
        """
        steps = self.parse(text)
        
        # Check if we have a final answer
        is_complete = any(step.final_answer for step in steps)
        
        return steps, is_complete
    
    def extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract just the final answer from ReAct response.
        
        Args:
            text: Raw response text
        
        Returns:
            Final answer string or None if not found
        """
        match = self.final_answer_pattern.search(text)
        return match.group(1).strip() if match else None
    
    def extract_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract tool calls from ReAct response.
        
        Args:
            text: Raw response text
        
        Returns:
            List of (tool_name, tool_args) tuples
        """
        steps = self.parse(text)
        tool_calls = []
        
        for step in steps:
            if step.action and step.action_input:
                tool_calls.append((step.action, step.action_input))
        
        return tool_calls
    
    def format_tools_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format available tools for inclusion in agent prompt.
        
        Args:
            tools: List of tool dictionaries with 'name' and 'description' keys
        
        Returns:
            Formatted tools string for prompt
        """
        if not tools:
            return "No tools available."
        
        tool_lines = ["Available tools:"]
        for tool in tools:
            name = tool.get('name', 'unknown')
            description = tool.get('description', 'No description')
            tool_lines.append(f"- **{name}**: {description}")
        
        return "\n".join(tool_lines)
    
    def create_react_prompt(
        self,
        question: str,
        tools: List[Dict[str, Any]],
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Create a ReAct prompt with tools and optional examples.
        
        Args:
            question: The question or task for the agent
            tools: List of available tools
            examples: Optional list of example ReAct responses
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a helpful AI assistant that can use tools to answer questions.",
            "Follow this format:",
            "",
            "Thought: I need to think about this step by step",
            "Action: tool_name",
            "Action Input: {\"arg1\": \"value1\", \"arg2\": \"value2\"}",
            "Observation: tool_result",
            "... (repeat if needed)",
            "Final Answer: final_response",
            "",
            self.format_tools_prompt(tools),
            "",
            "Question: " + question,
            "",
            "Let's approach this step by step:"
        ]
        
        if examples:
            prompt_parts.extend([
                "",
                "Examples:",
                *examples,
                "",
                "Now let's solve the question:"
            ])
        
        return "\n".join(prompt_parts) 