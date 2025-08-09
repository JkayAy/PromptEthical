# 🛡️ Ethical AI Prompt Library

A production-ready web application for testing LLM safety and robustness across multiple categories. Built with Python, Streamlit, and FastAPI, this tool helps researchers and developers evaluate AI model behavior through curated safety prompts.

## 🚀 Features

- **Comprehensive Prompt Categories**: Safety/Jailbreaks, Bias Tests, Truthfulness Checks, Reasoning Tests, and Prompt Injection Tests
- **Multi-Provider Support**: OpenAI (GPT), Anthropic (Claude), Cohere, and HuggingFace models
- **Real-time Execution**: Run prompts against models with live progress tracking
- **Historical Analysis**: Track model performance over time with interactive charts
- **Public API**: RESTful endpoints for programmatic access
- **Data Export**: Export results in CSV/JSON format for research
- **Production-Ready**: SQLite database, error handling, and rate limiting

## 📋 Prerequisites

- Python 3.8 or higher
- At least one LLM provider API key (OpenAI, Anthropic, Cohere, or HuggingFace)
- Modern web browser for the UI

## 🛠️ Installation & Setup

### Option 1: Quick Setup on Replit

1. **Fork this repository** on Replit
2. **Install dependencies** (automatic on Replit)
3. **Configure API keys**:
   - Copy `.env.example` to `.env`
   - Add your API keys from the providers you want to use
4. **Run the application**:
   - The Replit configuration will automatically start both Streamlit UI and FastAPI
   - Access the web interface at your Replit URL
   - API documentation available at `/docs`

### Option 2: Local Development

```bash
# Clone the repository
git clone <repository-url>
cd ethical-ai-prompt-library

# Install dependencies
pip install streamlit fastapi uvicorn pandas plotly requests python-dotenv openai anthropic

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize the database and prompts
python -c "from database import DatabaseManager; from prompts import PromptManager; DatabaseManager(); PromptManager()"

# Start the Streamlit UI
streamlit run app.py --server.port 5000

# In a separate terminal, start the API server
python api.py
