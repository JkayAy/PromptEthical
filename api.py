"""
FastAPI Public API Endpoints
Provides RESTful API access to the prompt library and execution functionality.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json

from database import DatabaseManager
from prompts import PromptManager, Prompt
from models import ModelManager
from runner import PromptRunner
from utils import format_timestamp

# Initialize FastAPI app
app = FastAPI(
    title="Ethical AI Prompt Library API",
    description="RESTful API for testing LLM safety and robustness",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
db_manager = DatabaseManager()
prompt_manager = PromptManager()
model_manager = ModelManager()
runner = PromptRunner(db_manager, model_manager)

# Pydantic models for API requests/responses
class PromptResponse(BaseModel):
    id: int
    category: str
    prompt: str
    description: str
    expected_behavior: str
    difficulty: str
    tags: List[str]
    created_at: str
    updated_at: str

class ExecutionRequest(BaseModel):
    prompt_ids: List[int] = Field(..., description="List of prompt IDs to execute")
    model_name: str = Field(..., description="Name of the model to use")
    max_tokens: Optional[int] = Field(500, description="Maximum tokens for model response")
    temperature: Optional[float] = Field(0.7, description="Model temperature")
    timeout: Optional[int] = Field(30, description="Timeout in seconds")

class ExecutionResponse(BaseModel):
    execution_id: str
    status: str
    message: str
    total_prompts: int
    estimated_time: float

class ExecutionResult(BaseModel):
    id: int
    prompt_id: int
    category: str
    model_name: str
    response: str
    status: str
    execution_time: float
    timestamp: str
    pass_fail_status: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    provider: str
    available: bool

class PromptCreate(BaseModel):
    category: str
    prompt: str
    description: str
    expected_behavior: str
    difficulty: str = "Medium"
    tags: List[str] = []

# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """API information and health check."""
    return {
        "name": "Ethical AI Prompt Library API",
        "version": "1.0.0",
        "description": "RESTful API for testing LLM safety and robustness",
        "endpoints": {
            "prompts": "/prompts",
            "models": "/models",
            "execute": "/execute",
            "results": "/results",
            "statistics": "/statistics",
            "documentation": "/docs"
        },
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Prompt endpoints
@app.get("/prompts", response_model=List[PromptResponse])
async def get_prompts(
    category: Optional[str] = Query(None, description="Filter by category"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
    search: Optional[str] = Query(None, description="Search in prompt text and description"),
    limit: Optional[int] = Query(100, description="Maximum number of prompts to return"),
    offset: Optional[int] = Query(0, description="Number of prompts to skip")
):
    """Get all prompts with optional filtering."""
    try:
        prompts_df = prompt_manager.get_prompts_dataframe()
        
        if prompts_df.empty:
            return []
        
        # Apply filters
        if category:
            prompts_df = prompts_df[prompts_df['category'] == category]
        
        if difficulty:
            prompts_df = prompts_df[prompts_df['difficulty'] == difficulty]
        
        if search:
            search_mask = (
                prompts_df['prompt'].astype(str).str.contains(search, case=False, na=False) |
                prompts_df['description'].astype(str).str.contains(search, case=False, na=False)
            )
            prompts_df = prompts_df[search_mask]
        
        # Apply pagination
        total_results = len(prompts_df)
        end_idx = min(offset + limit, len(prompts_df))
        prompts_df = prompts_df.iloc[offset:end_idx]
        
        # Convert to response format
        results = []
        for _, row in prompts_df.iterrows():
            results.append(PromptResponse(
                id=row['id'],
                category=row['category'],
                prompt=row['prompt'],
                description=row['description'],
                expected_behavior=row['expected_behavior'],
                difficulty=row['difficulty'],
                tags=row['tags'] if isinstance(row['tags'], list) else [],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prompts: {str(e)}")

@app.get("/prompts/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: int):
    """Get a specific prompt by ID."""
    try:
        prompt = prompt_manager.get_prompt_by_id(prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return PromptResponse(
            id=prompt.id,
            category=prompt.category,
            prompt=prompt.prompt,
            description=prompt.description,
            expected_behavior=prompt.expected_behavior,
            difficulty=prompt.difficulty,
            tags=prompt.tags,
            created_at=prompt.created_at,
            updated_at=prompt.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prompt: {str(e)}")

@app.post("/prompts", response_model=PromptResponse)
async def create_prompt(prompt_data: PromptCreate):
    """Create a new prompt."""
    try:
        new_prompt = prompt_manager.add_prompt(
            category=prompt_data.category,
            prompt_text=prompt_data.prompt,
            description=prompt_data.description,
            expected_behavior=prompt_data.expected_behavior,
            difficulty=prompt_data.difficulty,
            tags=prompt_data.tags
        )
        
        return PromptResponse(
            id=new_prompt.id,
            category=new_prompt.category,
            prompt=new_prompt.prompt,
            description=new_prompt.description,
            expected_behavior=new_prompt.expected_behavior,
            difficulty=new_prompt.difficulty,
            tags=new_prompt.tags,
            created_at=new_prompt.created_at,
            updated_at=new_prompt.updated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating prompt: {str(e)}")

@app.get("/prompts/categories", response_model=List[str])
async def get_categories():
    """Get all available prompt categories."""
    try:
        categories = prompt_manager.get_categories()
        return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

# Model endpoints
@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get all available models."""
    try:
        available_models = model_manager.get_available_models()
        
        results = []
        for model_name, model_info in available_models.items():
            results.append(ModelInfo(
                name=model_name,
                provider=model_info['provider'],
                available=model_info['available']
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving models: {str(e)}")

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        model_info = model_manager.get_model_info(model_name)
        if "error" in model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

# Execution endpoints
@app.post("/execute", response_model=ExecutionResponse)
async def execute_prompts(execution_request: ExecutionRequest, background_tasks: BackgroundTasks):
    """Execute prompts against a model."""
    try:
        # Validate prompt IDs
        prompts = prompt_manager.get_prompts_by_ids(execution_request.prompt_ids)
        if len(prompts) != len(execution_request.prompt_ids):
            missing_ids = set(execution_request.prompt_ids) - {p.id for p in prompts}
            raise HTTPException(
                status_code=400, 
                detail=f"Prompts not found: {list(missing_ids)}"
            )
        
        # Validate model
        available_models = model_manager.get_available_models()
        if execution_request.model_name not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {execution_request.model_name} not available"
            )
        
        # Generate execution ID
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(prompts)}"
        
        # Estimate execution time (rough estimate: 2 seconds per prompt)
        estimated_time = len(prompts) * 2.0
        
        # Start background execution
        background_tasks.add_task(
            execute_prompts_background,
            prompts,
            execution_request.model_name,
            execution_id,
            execution_request.max_tokens or 500,
            execution_request.temperature or 0.7,
            execution_request.timeout or 30
        )
        
        return ExecutionResponse(
            execution_id=execution_id,
            status="started",
            message="Execution started in background",
            total_prompts=len(prompts),
            estimated_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting execution: {str(e)}")

async def execute_prompts_background(prompts: List[Prompt], model_name: str, 
                                   execution_id: str, max_tokens: int, 
                                   temperature: float, timeout: int):
    """Background task for executing prompts."""
    try:
        # Execute all prompts
        results = runner.execute_batch_prompts(
            prompts=prompts,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        
        print(f"Execution {execution_id} completed: {len(results)} prompts processed")
        
    except Exception as e:
        print(f"Error in background execution {execution_id}: {str(e)}")

@app.get("/results", response_model=List[ExecutionResult])
async def get_results(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    days: Optional[int] = Query(30, description="Number of days to look back"),
    limit: Optional[int] = Query(100, description="Maximum number of results"),
    offset: Optional[int] = Query(0, description="Number of results to skip")
):
    """Get execution results with optional filtering."""
    try:
        results_df = db_manager.get_execution_results(
            days=days,
            model_name=model_name,
            category=category,
            status=status
        )
        
        if results_df.empty:
            return []
        
        # Apply pagination
        total_results = len(results_df)
        end_idx = min(offset + limit, len(results_df))
        results_df = results_df.iloc[offset:end_idx]
        
        # Convert to response format
        results = []
        for _, row in results_df.iterrows():
            results.append(ExecutionResult(
                id=row['id'],
                prompt_id=row['prompt_id'],
                category=row['category'],
                model_name=row['model_name'],
                response=row['response'],
                status=row['status'],
                execution_time=row['execution_time'],
                timestamp=row['timestamp'],
                pass_fail_status=row.get('pass_fail_status')
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@app.get("/results/{result_id}", response_model=ExecutionResult)
async def get_result(result_id: int):
    """Get a specific execution result by ID."""
    try:
        results_df = db_manager.get_execution_results()
        
        if results_df.empty:
            raise HTTPException(status_code=404, detail="Result not found")
        
        result_row = results_df[results_df['id'] == result_id]
        
        if result_row.empty:
            raise HTTPException(status_code=404, detail="Result not found")
        
        row = result_row.iloc[0]
        
        return ExecutionResult(
            id=row['id'],
            prompt_id=row['prompt_id'],
            category=row['category'],
            model_name=row['model_name'],
            response=row['response'],
            status=row['status'],
            execution_time=row['execution_time'],
            timestamp=row['timestamp'],
            pass_fail_status=row.get('pass_fail_status')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving result: {str(e)}")

# Statistics endpoints
@app.get("/statistics")
async def get_statistics(days: Optional[int] = Query(30, description="Number of days for statistics")):
    """Get execution statistics."""
    try:
        stats = runner.get_execution_statistics(days=days or 30)
        model_stats = db_manager.get_model_statistics(days=days or 30)
        category_stats = db_manager.get_category_performance(days=days or 30)
        db_stats = db_manager.get_database_stats()
        prompt_stats = prompt_manager.get_statistics()
        
        return {
            "execution_stats": stats,
            "model_stats": model_stats,
            "category_stats": category_stats,
            "database_stats": db_stats,
            "prompt_stats": prompt_stats,
            "date_range": f"Last {days} days",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.get("/statistics/models")
async def get_model_statistics(days: Optional[int] = Query(30, description="Number of days for statistics")):
    """Get detailed model performance statistics."""
    try:
        stats = db_manager.get_model_statistics(days=days or 30)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model statistics: {str(e)}")

@app.get("/statistics/categories")
async def get_category_statistics(days: Optional[int] = Query(30, description="Number of days for statistics")):
    """Get detailed category performance statistics."""
    try:
        stats = db_manager.get_category_performance(days=days or 30)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving category statistics: {str(e)}")

# Export endpoints
@app.get("/export/results")
async def export_results(
    format: str = Query("json", description="Export format: json or csv"),
    days: Optional[int] = Query(30, description="Number of days to export"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Export execution results."""
    try:
        results_df = db_manager.get_execution_results(
            days=days,
            model_name=model_name,
            category=category
        )
        
        if results_df.empty:
            raise HTTPException(status_code=404, detail="No results found for export")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == "csv":
            from utils import export_to_csv
            csv_data = export_to_csv(results_df)
            
            return JSONResponse(
                content={
                    "filename": f"ai_prompt_results_{timestamp}.csv",
                    "data": csv_data,
                    "format": "csv",
                    "total_records": len(results_df)
                }
            )
        
        elif format.lower() == "json":
            from utils import export_to_json
            json_data = export_to_json(results_df)
            
            return JSONResponse(
                content={
                    "filename": f"ai_prompt_results_{timestamp}.json",
                    "data": json.loads(json_data),
                    "format": "json",
                    "total_records": len(results_df)
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'csv'")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting results: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_stats = db_manager.get_database_stats()
        
        # Check available models
        available_models = model_manager.get_available_models()
        
        # Check prompt library
        prompt_stats = prompt_manager.get_statistics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if "error" not in db_stats else "error",
            "models_available": len(available_models),
            "prompts_loaded": prompt_stats["total_prompts"],
            "version": "1.0.0"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "version": "1.0.0"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
