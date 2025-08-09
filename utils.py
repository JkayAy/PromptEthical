"""
Utility Functions
Shared helper functions for the Ethical AI Prompt Library.
"""

import json
import csv
import io
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import re
import hashlib
import os

def format_timestamp(timestamp: Union[str, datetime], format_type: str = "display") -> str:
    """
    Format timestamp for display or storage.
    
    Args:
        timestamp: Timestamp as string or datetime object
        format_type: "display", "filename", or "iso"
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = timestamp
    
    if format_type == "display":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "filename":
        return dt.strftime("%Y%m%d_%H%M%S")
    elif format_type == "iso":
        return dt.isoformat()
    else:
        return str(dt)

def export_to_csv(dataframe: pd.DataFrame) -> str:
    """
    Export pandas DataFrame to CSV string.
    
    Args:
        dataframe: Pandas DataFrame to export
        
    Returns:
        CSV data as string
    """
    output = io.StringIO()
    dataframe.to_csv(output, index=False)
    return output.getvalue()

def export_to_json(dataframe: pd.DataFrame) -> str:
    """
    Export pandas DataFrame to JSON string.
    
    Args:
        dataframe: Pandas DataFrame to export
        
    Returns:
        JSON data as string
    """
    # Convert DataFrame to dict and handle any date/time columns
    data = dataframe.to_dict('records')
    
    # Handle any special data types
    for record in data:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                record[key] = value.isoformat()
    
    return json.dumps(data, indent=2, ensure_ascii=False)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove extra spaces and dots
    sanitized = re.sub(r'[\s.]+', '_', sanitized)
    
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    return sanitized

def calculate_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def validate_json_structure(data: Any, required_fields: List[str]) -> List[str]:
    """
    Validate JSON data structure.
    
    Args:
        data: JSON data to validate
        required_fields: List of required field names
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Data must be a JSON object")
        return errors
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None or data[field] == "":
            errors.append(f"Field '{field}' cannot be empty")
    
    return errors

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of result
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for tagging/search.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    # Simple keyword extraction - can be enhanced with NLP libraries
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    # Filter stop words and get unique keywords
    keywords = [word for word in set(words) if word not in stop_words]
    
    # Sort by frequency (simple approach)
    word_freq = {word: words.count(word) for word in keywords}
    sorted_keywords = sorted(keywords, key=lambda x: word_freq[x], reverse=True)
    
    return sorted_keywords[:max_keywords]

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def parse_model_response_stats(usage_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and standardize model response usage statistics.
    
    Args:
        usage_data: Raw usage data from model response
        
    Returns:
        Standardized usage statistics
    """
    if not usage_data:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_estimate": 0.0
        }
    
    # Standardize different provider formats
    input_tokens = (
        usage_data.get("prompt_tokens", 0) or
        usage_data.get("input_tokens", 0) or
        0
    )
    
    output_tokens = (
        usage_data.get("completion_tokens", 0) or
        usage_data.get("output_tokens", 0) or
        usage_data.get("tokens", 0) or
        0
    )
    
    total_tokens = (
        usage_data.get("total_tokens", 0) or
        input_tokens + output_tokens
    )
    
    # Rough cost estimation (varies by provider and model)
    # These are approximate rates and should be updated based on actual pricing
    cost_per_1k_input = 0.0015  # ~$1.50 per 1K input tokens (GPT-4 rate)
    cost_per_1k_output = 0.002  # ~$2.00 per 1K output tokens
    
    cost_estimate = (
        (input_tokens / 1000) * cost_per_1k_input +
        (output_tokens / 1000) * cost_per_1k_output
    )
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_estimate": round(cost_estimate, 6)
    }

def analyze_prompt_complexity(prompt_text: str) -> Dict[str, Any]:
    """
    Analyze prompt complexity and characteristics.
    
    Args:
        prompt_text: The prompt text to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    # Basic text statistics
    word_count = len(prompt_text.split())
    char_count = len(prompt_text)
    sentence_count = len(re.split(r'[.!?]+', prompt_text))
    
    # Calculate readability (simple approach)
    avg_words_per_sentence = word_count / max(sentence_count, 1)
    avg_chars_per_word = char_count / max(word_count, 1)
    
    # Complexity indicators
    has_questions = '?' in prompt_text
    has_instructions = any(word in prompt_text.lower() for word in [
        'please', 'tell me', 'explain', 'describe', 'list', 'generate'
    ])
    has_constraints = any(word in prompt_text.lower() for word in [
        'without', "don't", 'avoid', 'except', 'only', 'must', 'should'
    ])
    
    # Complexity score (0-10)
    complexity_score = min(10, (
        (word_count / 50) * 2 +  # Length factor
        (avg_words_per_sentence / 15) * 2 +  # Sentence complexity
        (2 if has_questions else 0) +  # Questions add complexity
        (2 if has_instructions else 0) +  # Instructions add complexity
        (2 if has_constraints else 0)  # Constraints add complexity
    ))
    
    return {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "avg_chars_per_word": round(avg_chars_per_word, 1),
        "has_questions": has_questions,
        "has_instructions": has_instructions,
        "has_constraints": has_constraints,
        "complexity_score": round(complexity_score, 1),
        "complexity_level": (
            "Simple" if complexity_score < 3 else
            "Moderate" if complexity_score < 6 else
            "Complex" if complexity_score < 8 else
            "Very Complex"
        )
    }

def generate_execution_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from execution results.
    
    Args:
        results: List of execution result dictionaries
        
    Returns:
        Summary statistics
    """
    if not results:
        return {
            "total_executions": 0,
            "success_rate": 0,
            "average_execution_time": 0,
            "total_execution_time": 0,
            "models_used": [],
            "categories_tested": [],
            "pass_rate": 0
        }
    
    total_executions = len(results)
    successful = sum(1 for r in results if r.get('status') == 'success')
    success_rate = (successful / total_executions) * 100
    
    execution_times = [r.get('execution_time', 0) for r in results]
    avg_execution_time = sum(execution_times) / len(execution_times)
    total_execution_time = sum(execution_times)
    
    models_used = list(set(r.get('model_name', '') for r in results))
    categories_tested = list(set(r.get('category', '') for r in results))
    
    # Calculate pass rate for tests with pass/fail status
    pass_fail_results = [r for r in results if r.get('pass_fail_status')]
    if pass_fail_results:
        passed = sum(1 for r in pass_fail_results if r.get('pass_fail_status') == 'pass')
        pass_rate = (passed / len(pass_fail_results)) * 100
    else:
        pass_rate = 0
    
    return {
        "total_executions": total_executions,
        "success_rate": round(success_rate, 1),
        "average_execution_time": round(avg_execution_time, 2),
        "total_execution_time": round(total_execution_time, 2),
        "models_used": models_used,
        "categories_tested": categories_tested,
        "pass_rate": round(pass_rate, 1)
    }

def validate_api_key(api_key: str, provider: str) -> bool:
    """
    Basic API key validation.
    
    Args:
        api_key: API key to validate
        provider: Provider name (openai, anthropic, cohere, huggingface)
        
    Returns:
        True if key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format validation based on provider
    if provider.lower() == "openai":
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider.lower() == "anthropic":
        return api_key.startswith("sk-ant-") and len(api_key) > 20
    elif provider.lower() == "cohere":
        return len(api_key) > 20  # Cohere keys don't have standard prefix
    elif provider.lower() == "huggingface":
        return api_key.startswith("hf_") and len(api_key) > 20
    
    return len(api_key) > 10  # Generic validation

def create_backup_filename(base_name: str, extension: str = ".bak") -> str:
    """
    Create timestamped backup filename.
    
    Args:
        base_name: Base filename
        extension: File extension for backup
        
    Returns:
        Backup filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_without_ext = os.path.splitext(base_name)[0]
    return f"{name_without_ext}_{timestamp}{extension}"

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON data or default value
    """
    try:
        return json.loads(json_string) if json_string else default
    except (json.JSONDecodeError, TypeError):
        return default

def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information for debugging.
    
    Returns:
        Dictionary with environment details
    """
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "timestamp": datetime.now().isoformat(),
        "working_directory": os.getcwd(),
        "environment_variables": {
            key: "***" if "key" in key.lower() or "secret" in key.lower() else value
            for key, value in os.environ.items()
            if not key.lower().startswith("secret")
        }
    }
