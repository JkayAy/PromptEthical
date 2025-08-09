# Overview

The Ethical AI Prompt Library is a production-ready web application designed to test LLM safety and robustness across multiple categories. The application provides researchers and developers with a comprehensive platform to evaluate AI model behavior using curated prompts for safety stress-testing, bias detection, truthfulness checks, reasoning evaluation, and prompt injection robustness testing.

The system combines a user-friendly Streamlit web interface with a RESTful FastAPI backend, enabling both interactive testing and programmatic access. It supports multiple LLM providers (OpenAI, Anthropic, Cohere, HuggingFace) and includes features for historical analysis, data export, and real-time execution tracking.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit-based UI**: Multi-tab interface with tabs for Prompt Library, Run Tests, Results, and Analysis
- **Interactive Components**: Real-time progress tracking, interactive charts using Plotly, and data export functionality
- **Responsive Design**: Wide layout configuration optimized for data visualization and analysis

## Backend Architecture
- **FastAPI REST API**: Provides programmatic access with comprehensive endpoints for prompts and execution
- **Modular Design**: Separated concerns across dedicated modules:
  - `database.py`: SQLite database operations
  - `prompts.py`: Prompt management and library operations
  - `models.py`: LLM provider integrations with abstracted interfaces
  - `runner.py`: Prompt execution engine with error handling
  - `utils.py`: Shared utility functions

## Data Storage Solutions
- **SQLite Database**: Local file-based storage with two main tables:
  - `prompts`: Stores prompt library with categories, descriptions, and metadata
  - `execution_results`: Logs all test executions with performance metrics
- **JSON Configuration**: Prompt library stored in structured JSON format for easy management
- **File-based Exports**: CSV and JSON export capabilities for research data

## Authentication and Authorization
- **API Key Management**: Environment variable-based configuration for multiple LLM providers
- **Rate Limiting**: Built-in request throttling to respect API limits
- **CORS Support**: Configured for cross-origin requests in the FastAPI layer

## Execution Engine
- **Asynchronous Processing**: Background task execution with queue management
- **Multi-Provider Support**: Abstracted model interface supporting OpenAI, Anthropic, Cohere, and HuggingFace
- **Error Handling**: Comprehensive exception handling with timeout management
- **Result Logging**: Automatic storage of all execution results with performance metrics

## Data Management
- **Prompt Categories**: Organized into Safety/Jailbreaks, Bias Tests, Truthfulness Checks, Reasoning Tests, and Prompt Injection Tests
- **Metadata Tracking**: Difficulty levels, tags, timestamps, and expected behaviors
- **Historical Analysis**: Time-series tracking of model performance across different categories

# External Dependencies

## LLM Provider APIs
- **OpenAI API**: GPT model integration with official Python client
- **Anthropic API**: Claude model support with official SDK
- **Cohere API**: Cohere model integration via REST API
- **HuggingFace Inference API**: Access to various HuggingFace models

## Core Python Frameworks
- **Streamlit**: Web application framework for the user interface
- **FastAPI**: Modern web framework for the REST API backend
- **SQLite**: Embedded database for local data storage
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualization and charting

## Supporting Libraries
- **Pydantic**: Data validation and API request/response models
- **Uvicorn**: ASGI server for FastAPI deployment
- **python-dotenv**: Environment variable management
- **asyncio**: Asynchronous programming support
- **requests**: HTTP client for external API calls

## Development and Deployment
- **Replit Platform**: Cloud-based development and hosting environment
- **Environment Configuration**: `.env` file-based API key management
- **Automatic Dependency Management**: pip-based package installation
- **CORS Middleware**: Cross-origin request handling for API access