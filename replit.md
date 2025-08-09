# Overview

The Ethical AI Prompt Library is an enterprise-grade AI safety research platform designed to attract top-tier talent from companies like Meta, Anthropic, and Google. This production-ready application demonstrates advanced ML engineering capabilities through comprehensive LLM safety testing, bias detection, adversarial prompt generation, and sophisticated anomaly detection systems.

The platform combines cutting-edge research capabilities with production-grade monitoring, featuring automated safety classification using ML models, real-time performance optimization, enterprise dashboards for executive reporting, and advanced analytics suitable for research publications. It showcases expertise in both AI safety research and large-scale system architecture.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Advanced Multi-Tab Interface**: Six specialized dashboards including Enterprise Executive Overview and Research Analytics
- **Real-time Monitoring**: Production-grade alerting system with performance metrics and anomaly detection
- **Interactive Visualizations**: Sophisticated Plotly charts for model comparison matrices, performance trends, and research insights
- **Enterprise Dashboards**: Executive-level KPIs, technical metrics for engineers, and research analytics for AI scientists

## Backend Architecture
- **Production API Layer**: FastAPI with comprehensive endpoints, auto-scaling recommendations, and performance optimization
- **Advanced ML Pipeline**: 
  - `ml_features.py`: Production ML safety classifier with feature engineering and anomaly detection
  - `advanced_analysis.py`: Sophisticated toxicity analysis, bias detection, and adversarial prompt generation
  - `enterprise_dashboard.py`: Executive and technical monitoring dashboards with real-time alerting
- **Core Infrastructure**: Database operations, model abstractions, execution engine, and utility functions
- **Research Capabilities**: Statistical analysis, benchmark generation, and publication-ready reporting

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
- **scikit-learn**: Production ML models for safety classification and anomaly detection
- **scipy**: Statistical analysis and significance testing for research-grade metrics

## Development and Deployment
- **Replit Platform**: Cloud-based development and hosting environment
- **Environment Configuration**: `.env` file-based API key management
- **Automatic Dependency Management**: pip-based package installation
- **CORS Middleware**: Cross-origin request handling for API access