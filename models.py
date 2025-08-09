"""
Model Management Module
Handles API integrations with different LLM providers.
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests
import json
from abc import ABC, abstractmethod

# Import API clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

@dataclass
class ModelResponse:
    """Standardized response format for all model providers."""
    text: str
    usage: Dict[str, Any]
    model: str
    provider: str
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0

class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    @abstractmethod
    def call_model(self, prompt: str, **kwargs) -> ModelResponse:
        """Call the model with given prompt and parameters."""
        pass
    
    def _rate_limit(self):
        """Implement basic rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if OpenAI is None:
            raise ImportError("OpenAI package not available")
        self.client = OpenAI(api_key=api_key)
        self.provider_name = "OpenAI"
    
    def call_model(self, prompt: str, model: str = "gpt-4o", **kwargs) -> ModelResponse:
        """
        Call OpenAI model with the given prompt.
        
        Args:
            prompt: The input prompt
            model: Model name (default: gpt-4o - the newest OpenAI model)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
        """
        self._rate_limit()
        start_time = time.time()
        
        try:
            # Extract parameters
            max_tokens = kwargs.get('max_tokens', 500)
            temperature = kwargs.get('temperature', 0.7)
            timeout = kwargs.get('timeout', 30)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            return ModelResponse(
                text=response.choices[0].message.content or "",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                model=model,
                provider=self.provider_name,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ModelResponse(
                text="",
                usage={},
                model=model,
                provider=self.provider_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class AnthropicProvider(BaseModelProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if Anthropic is None:
            raise ImportError("Anthropic package not available")
        self.client = Anthropic(api_key=api_key)
        self.provider_name = "Anthropic"
    
    def call_model(self, prompt: str, model: str = "claude-sonnet-4-20250514", **kwargs) -> ModelResponse:
        """
        Call Anthropic model with the given prompt.
        
        Args:
            prompt: The input prompt
            model: Model name (default: claude-sonnet-4-20250514 - the newest Anthropic model)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
        """
        self._rate_limit()
        start_time = time.time()
        
        try:
            # Extract parameters
            max_tokens = kwargs.get('max_tokens', 500)
            temperature = kwargs.get('temperature', 0.7)
            timeout = kwargs.get('timeout', 30)
            
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            return ModelResponse(
                text=response.content[0].text if response.content and len(response.content) > 0 else "",
                usage={
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
                },
                model=model,
                provider=self.provider_name,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ModelResponse(
                text="",
                usage={},
                model=model,
                provider=self.provider_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class CohereProvider(BaseModelProvider):
    """Cohere API provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.provider_name = "Cohere"
        self.base_url = "https://api.cohere.ai/v1"
    
    def call_model(self, prompt: str, model: str = "command-r", **kwargs) -> ModelResponse:
        """
        Call Cohere model with the given prompt.
        
        Args:
            prompt: The input prompt
            model: Model name (default: command-r)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
        """
        self._rate_limit()
        start_time = time.time()
        
        try:
            # Extract parameters
            max_tokens = kwargs.get('max_tokens', 500)
            temperature = kwargs.get('temperature', 0.7)
            timeout = kwargs.get('timeout', 30)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "message": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            execution_time = time.time() - start_time
            
            return ModelResponse(
                text=result["text"],
                usage={
                    "input_tokens": result.get("meta", {}).get("tokens", {}).get("input_tokens", 0),
                    "output_tokens": result.get("meta", {}).get("tokens", {}).get("output_tokens", 0),
                    "total_tokens": result.get("meta", {}).get("tokens", {}).get("input_tokens", 0) + 
                                   result.get("meta", {}).get("tokens", {}).get("output_tokens", 0)
                },
                model=model,
                provider=self.provider_name,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ModelResponse(
                text="",
                usage={},
                model=model,
                provider=self.provider_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class HuggingFaceProvider(BaseModelProvider):
    """HuggingFace Inference API provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.provider_name = "HuggingFace"
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def call_model(self, prompt: str, model: str = "microsoft/DialoGPT-large", **kwargs) -> ModelResponse:
        """
        Call HuggingFace model with the given prompt.
        
        Args:
            prompt: The input prompt
            model: Model name (default: microsoft/DialoGPT-large)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
        """
        self._rate_limit()
        start_time = time.time()
        
        try:
            # Extract parameters
            max_tokens = kwargs.get('max_tokens', 500)
            temperature = kwargs.get('temperature', 0.7)
            timeout = kwargs.get('timeout', 30)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            execution_time = time.time() - start_time
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                text = result.get("generated_text", "")
            else:
                text = str(result)
            
            return ModelResponse(
                text=text,
                usage={"tokens": len(text.split())},  # Approximate token count
                model=model,
                provider=self.provider_name,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ModelResponse(
                text="",
                usage={},
                model=model,
                provider=self.provider_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class ModelManager:
    """Main class for managing multiple model providers."""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on API keys."""
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and OpenAI is not None:
            try:
                self.providers["gpt-4o"] = OpenAIProvider(openai_key)
                self.providers["gpt-4"] = OpenAIProvider(openai_key)
                self.providers["gpt-3.5-turbo"] = OpenAIProvider(openai_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI provider: {e}")
        
        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and Anthropic is not None:
            try:
                self.providers["claude-sonnet-4-20250514"] = AnthropicProvider(anthropic_key)
                self.providers["claude-3-7-sonnet-20250219"] = AnthropicProvider(anthropic_key)
                self.providers["claude-3-5-sonnet-20241022"] = AnthropicProvider(anthropic_key)
            except Exception as e:
                print(f"Failed to initialize Anthropic provider: {e}")
        
        # Cohere
        cohere_key = os.getenv("COHERE_API_KEY")
        if cohere_key:
            try:
                self.providers["command-r"] = CohereProvider(cohere_key)
                self.providers["command-r-plus"] = CohereProvider(cohere_key)
            except Exception as e:
                print(f"Failed to initialize Cohere provider: {e}")
        
        # HuggingFace
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            try:
                self.providers["microsoft/DialoGPT-large"] = HuggingFaceProvider(hf_key)
                self.providers["google/flan-t5-large"] = HuggingFaceProvider(hf_key)
                self.providers["meta-llama/Llama-2-7b-chat-hf"] = HuggingFaceProvider(hf_key)
            except Exception as e:
                print(f"Failed to initialize HuggingFace provider: {e}")
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models with their provider information."""
        models = {}
        for model_name, provider in self.providers.items():
            models[model_name] = {
                "provider": provider.provider_name,
                "available": True
            }
        return models
    
    def call_model(self, model_name: str, prompt: str, **kwargs) -> ModelResponse:
        """
        Call a specific model with the given prompt.
        
        Args:
            model_name: Name of the model to call
            prompt: The input prompt
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            ModelResponse object with the result
        """
        if model_name not in self.providers:
            return ModelResponse(
                text="",
                usage={},
                model=model_name,
                provider="Unknown",
                success=False,
                error=f"Model {model_name} not available"
            )
        
        provider = self.providers[model_name]
        
        # Extract model-specific name if needed
        if provider.provider_name == "OpenAI":
            actual_model = model_name
        elif provider.provider_name == "Anthropic":
            actual_model = model_name
        elif provider.provider_name == "Cohere":
            actual_model = model_name
        elif provider.provider_name == "HuggingFace":
            actual_model = model_name
        else:
            actual_model = model_name
        
        return provider.call_model(prompt, model=actual_model, **kwargs)
    
    def test_model_availability(self, model_name: str) -> bool:
        """Test if a model is available and responding."""
        if model_name not in self.providers:
            return False
        
        try:
            test_prompt = "Hello, this is a test."
            response = self.call_model(model_name, test_prompt, max_tokens=10)
            return response.success
        except Exception:
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        if model_name not in self.providers:
            return {"error": "Model not found"}
        
        provider = self.providers[model_name]
        return {
            "model_name": model_name,
            "provider": provider.provider_name,
            "available": self.test_model_availability(model_name),
            "rate_limit_delay": provider.rate_limit_delay
        }
