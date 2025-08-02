"""
ReAct (Reasoning and Acting) agent implementation with streaming and tool integration.
"""

import asyncio
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from ..llms.openai_client import Message, OpenAIClient
from .parser import ReActParser, ReActStep
from .tool_registry import ToolRegistry, get_tool_registry


class AgentConfig(BaseModel):
    """Configuration for ReAct agent."""
    
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    enable_streaming: bool = True
    timeout: Optional[float] = 30.0
    retry_on_error: bool = True
    max_retries: int = 3


class AgentState(BaseModel):
    """State maintained by the agent during execution."""
    
    question: str
    conversation_history: List[Message] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    current_iteration: int = 0
    final_answer: Optional[str] = None
    error: Optional[str] = None


class ReActAgent:
    """
    ReAct (Reasoning and Acting) agent with streaming support and tool integration.
    
    Implements the ReAct pattern: Thought -> Action -> Observation -> Final Answer
    """
    
    def __init__(
        self,
        llm: OpenAIClient,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None
    ):
        self.llm = llm
        self.tool_registry = tool_registry or get_tool_registry()
        self.config = config or AgentConfig()
        self.parser = ReActParser()
        
        # Execution tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = 0.0
    
    async def run(
        self,
        question: str,
        streaming_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Run the agent to answer a question using tools.
        
        Args:
            question: The question to answer
            streaming_callback: Optional callback for streaming partial responses
        
        Returns:
            Final answer string
        """
        start_time = time.time()
        self._execution_count += 1
        
        # Initialize agent state
        state = AgentState(question=question)
        
        try:
            # Main ReAct loop
            for iteration in range(self.config.max_iterations):
                state.current_iteration = iteration
                
                # Generate agent response
                response = await self._generate_response(state, streaming_callback)
                
                # Parse the response
                steps = self.parser.parse(response)
                
                # Process each step
                for step in steps:
                    if step.final_answer:
                        state.final_answer = step.final_answer
                        return step.final_answer
                    
                    if step.action and step.action_input:
                        # Execute tool
                        try:
                            result = await self.tool_registry.execute_tool(
                                step.action, 
                                **step.action_input
                            )
                            
                            # Add to conversation history
                            state.conversation_history.extend([
                                Message(role="assistant", content=response),
                                Message(role="user", content=f"Observation: {result}")
                            ])
                            
                            # Store tool result
                            state.tool_results.append({
                                "action": step.action,
                                "input": step.action_input,
                                "result": result,
                                "iteration": iteration
                            })
                            
                        except Exception as e:
                            error_msg = f"Tool '{step.action}' failed: {str(e)}"
                            state.error = error_msg
                            
                            # Add error to conversation history
                            state.conversation_history.extend([
                                Message(role="assistant", content=response),
                                Message(role="user", content=f"Observation: {error_msg}")
                            ])
                            
                            if not self.config.retry_on_error:
                                raise e
            
            # Max iterations reached
            raise RuntimeError(f"Agent exceeded maximum iterations ({self.config.max_iterations})")
            
        except Exception as e:
            state.error = str(e)
            raise
        finally:
            # Update metrics
            self._last_execution_time = time.time() - start_time
            self._total_execution_time += self._last_execution_time
    
    async def run_streaming(
        self,
        question: str,
        callback: Callable[[str, bool], None]
    ) -> str:
        """
        Run the agent with streaming support.
        
        Args:
            question: The question to answer
            callback: Callback function called with (chunk, is_complete)
        
        Returns:
            Final answer string
        """
        start_time = time.time()
        self._execution_count += 1
        
        # Initialize agent state
        state = AgentState(question=question)
        full_response = ""
        
        try:
            # Main ReAct loop
            for iteration in range(self.config.max_iterations):
                state.current_iteration = iteration
                
                # Generate streaming response
                async for chunk in self._generate_streaming_response(state):
                    full_response += chunk
                    # Handle callback properly (could be sync or async)
                    if asyncio.iscoroutinefunction(callback):
                        await callback(chunk, False)
                    else:
                        callback(chunk, False)
                    
                    # Check if we have a complete step
                    steps, is_complete = self.parser.parse_streaming(full_response)
                    
                    if is_complete:
                        # Process the complete response
                        for step in steps:
                            if step.final_answer:
                                state.final_answer = step.final_answer
                                # Handle callback properly
                                if asyncio.iscoroutinefunction(callback):
                                    await callback("", True)  # Signal completion
                                else:
                                    callback("", True)  # Signal completion
                                return step.final_answer
                            
                            if step.action and step.action_input:
                                # Execute tool
                                try:
                                    result = await self.tool_registry.execute_tool(
                                        step.action, 
                                        **step.action_input
                                    )
                                    
                                    # Add to conversation history
                                    state.conversation_history.extend([
                                        Message(role="assistant", content=full_response),
                                        Message(role="user", content=f"Observation: {result}")
                                    ])
                                    
                                    # Store tool result
                                    state.tool_results.append({
                                        "action": step.action,
                                        "input": step.action_input,
                                        "result": result,
                                        "iteration": iteration
                                    })
                                    
                                    # Reset for next iteration
                                    full_response = ""
                                    break
                                    
                                except Exception as e:
                                    error_msg = f"Tool '{step.action}' failed: {str(e)}"
                                    state.error = error_msg
                                    
                                    # Add error to conversation history
                                    state.conversation_history.extend([
                                        Message(role="assistant", content=full_response),
                                        Message(role="user", content=f"Observation: {error_msg}")
                                    ])
                                    
                                    if not self.config.retry_on_error:
                                        raise e
                                    
                                    # Reset for next iteration
                                    full_response = ""
                                    break
            
            # Max iterations reached
            raise RuntimeError(f"Agent exceeded maximum iterations ({self.config.max_iterations})")
            
        except Exception as e:
            state.error = str(e)
            raise
        finally:
            # Update metrics
            self._last_execution_time = time.time() - start_time
            self._total_execution_time += self._last_execution_time
    
    async def _generate_response(self, state: AgentState, streaming_callback: Optional[Callable] = None) -> str:
        """Generate agent response using LLM."""
        # Build conversation history
        messages = self._build_messages(state)
        
        # Get response from LLM
        if self.config.enable_streaming and streaming_callback:
            response = ""
            async for chunk in self.llm.stream(messages, self.config.temperature, self.config.max_tokens):
                response += chunk
                # Handle callback properly (could be sync or async)
                if asyncio.iscoroutinefunction(streaming_callback):
                    await streaming_callback(chunk)
                else:
                    streaming_callback(chunk)
            return response
        else:
            llm_response = await self.llm.chat(messages, self.config.temperature, self.config.max_tokens)
            return llm_response.content
    
    async def _generate_streaming_response(self, state: AgentState) -> AsyncGenerator[str, None]:
        """Generate streaming agent response using LLM."""
        messages = self._build_messages(state)
        
        try:
            async for chunk in self.llm.stream(messages, self.config.temperature, self.config.max_tokens):
                yield chunk
        except Exception as e:
            # Ensure proper cleanup of async generator
            raise e
    
    def _build_messages(self, state: AgentState) -> List[Message]:
        """Build conversation messages for the LLM."""
        messages = []
        
        # System message with tools
        tools = self.tool_registry.list_tools()
        system_prompt = self.parser.create_react_prompt(state.question, tools)
        messages.append(Message(role="system", content=system_prompt))
        
        # Add conversation history
        messages.extend(state.conversation_history)
        
        return messages
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent execution metrics."""
        return {
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "last_execution_time": self._last_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._execution_count 
                if self._execution_count > 0 else 0
            ),
            "tool_metrics": self.tool_registry.get_execution_metrics()
        }
    
    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = 0.0
        self.tool_registry.clear_history() 