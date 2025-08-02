"""
OpenAI client adapter with streaming support.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import httpx
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message for LLM communication."""
    
    role: str
    content: str


class LLMResponse(BaseModel):
    """Response from LLM with metadata."""
    
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    model: Optional[str] = None


class OpenAIClient:
    """
    Async OpenAI client with streaming support.
    
    Supports both chat completions and streaming responses.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Use environment variable if no API key provided
        if not self.api_key:
            import os
            # Try to load .env file
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # python-dotenv not installed, continue without it
            
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    async def _make_sync_request(self, endpoint: str, data: Dict[str, Any]) -> httpx.Response:
        """Make synchronous HTTP request to OpenAI API with retries."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=data)
                    response.raise_for_status()
                    return response
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.max_retries:
                    # Rate limit - wait and retry
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                raise
        
        raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts")
    
    async def _make_streaming_request(self, endpoint: str, data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Make streaming HTTP request to OpenAI API with retries."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("POST", url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.strip():
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    if data_str == "[DONE]":
                                        return  # Proper exit
                                    try:
                                        yield json.loads(data_str)
                                    except json.JSONDecodeError:
                                        continue
                return  # Success, exit generator
                        
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.max_retries:
                    # Rate limit - wait and retry
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                raise
        
        raise RuntimeError(f"Streaming request failed after {self.max_retries + 1} attempts")
    
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Send chat completion request.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        
        Returns:
            LLMResponse or async generator for streaming
        """
        data = {
            "model": self.model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature,
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        if stream:
            return self._stream_chat(data)
        else:
            return await self._chat(data)
    
    async def _chat(self, data: Dict[str, Any]) -> LLMResponse:
        """Non-streaming chat completion."""
        response = await self._make_sync_request("chat/completions", data)
        result = response.json()
        
        choice = result["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            usage=result.get("usage"),
            finish_reason=choice.get("finish_reason"),
            model=result.get("model")
        )
    
    async def _stream_chat(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streaming chat completion."""
        data["stream"] = True
        
        async for chunk in self._make_streaming_request("chat/completions", data):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
    
    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion response.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Yields:
            Response content chunks
        """
        async for chunk in self._stream_chat({
            "model": self.model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature,
            "stream": True,
            **({"max_tokens": max_tokens} if max_tokens else {})
        }):
            yield chunk 