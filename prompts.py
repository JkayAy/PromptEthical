"""
Prompt Management Module
Handles loading, storing, and managing the prompt library.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

@dataclass
class Prompt:
    """Data class representing a single prompt."""
    id: int
    category: str
    prompt: str
    description: str
    expected_behavior: str
    difficulty: str
    tags: List[str]
    created_at: str
    updated_at: str

class PromptManager:
    """Manages the prompt library and provides search/filter functionality."""
    
    def __init__(self, prompts_file: str = "data/prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = []
        self.load_prompts()
    
    def load_prompts(self):
        """Load prompts from JSON file."""
        try:
            if os.path.exists(self.prompts_file):
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.prompts = [Prompt(**prompt_data) for prompt_data in data['prompts']]
            else:
                print(f"Prompts file {self.prompts_file} not found. Starting with empty library.")
                self.prompts = []
        except Exception as e:
            print(f"Error loading prompts: {e}")
            self.prompts = []
    
    def save_prompts(self):
        """Save prompts to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.prompts_file), exist_ok=True)
            
            data = {
                "version": "1.0",
                "description": "Ethical AI Prompt Library for testing LLM safety and robustness",
                "categories": self.get_categories(),
                "prompts": [
                    {
                        "id": prompt.id,
                        "category": prompt.category,
                        "prompt": prompt.prompt,
                        "description": prompt.description,
                        "expected_behavior": prompt.expected_behavior,
                        "difficulty": prompt.difficulty,
                        "tags": prompt.tags,
                        "created_at": prompt.created_at,
                        "updated_at": prompt.updated_at
                    }
                    for prompt in self.prompts
                ]
            }
            
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving prompts: {e}")
    
    def get_prompts_dataframe(self) -> pd.DataFrame:
        """Convert prompts to pandas DataFrame for easy display and analysis."""
        if not self.prompts:
            return pd.DataFrame()
        
        data = []
        for prompt in self.prompts:
            data.append({
                'id': prompt.id,
                'category': prompt.category,
                'prompt': prompt.prompt,
                'description': prompt.description,
                'expected_behavior': prompt.expected_behavior,
                'difficulty': prompt.difficulty,
                'tags': prompt.tags,
                'created_at': prompt.created_at,
                'updated_at': prompt.updated_at
            })
        
        return pd.DataFrame(data)
    
    def get_categories(self) -> List[str]:
        """Get list of all unique categories."""
        return list(set(prompt.category for prompt in self.prompts))
    
    def get_prompts_by_category(self, category: str) -> List[Prompt]:
        """Get all prompts in a specific category."""
        return [prompt for prompt in self.prompts if prompt.category == category]
    
    def get_prompts_by_ids(self, prompt_ids: List[int]) -> List[Prompt]:
        """Get prompts by their IDs."""
        id_set = set(prompt_ids)
        return [prompt for prompt in self.prompts if prompt.id in id_set]
    
    def search_prompts(self, query: str, categories: Optional[List[str]] = None) -> List[Prompt]:
        """Search prompts by text query and optionally filter by categories."""
        results = []
        query_lower = query.lower()
        
        for prompt in self.prompts:
            # Category filter
            if categories and prompt.category not in categories:
                continue
            
            # Text search in prompt, description, and tags
            if (query_lower in prompt.prompt.lower() or
                query_lower in prompt.description.lower() or
                any(query_lower in tag.lower() for tag in prompt.tags)):
                results.append(prompt)
        
        return results
    
    def add_prompt(self, category: str, prompt_text: str, description: str,
                   expected_behavior: str, difficulty: str = "Medium",
                   tags: Optional[List[str]] = None) -> Prompt:
        """Add a new prompt to the library."""
        from datetime import datetime
        
        if tags is None:
            tags = []
        
        # Generate new ID
        max_id = max((p.id for p in self.prompts), default=0)
        new_id = max_id + 1
        
        timestamp = datetime.now().isoformat()
        
        new_prompt = Prompt(
            id=new_id,
            category=category,
            prompt=prompt_text,
            description=description,
            expected_behavior=expected_behavior,
            difficulty=difficulty,
            tags=tags,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        self.prompts.append(new_prompt)
        self.save_prompts()
        
        return new_prompt
    
    def update_prompt(self, prompt_id: int, **kwargs) -> Optional[Prompt]:
        """Update an existing prompt."""
        from datetime import datetime
        
        for i, prompt in enumerate(self.prompts):
            if prompt.id == prompt_id:
                # Update fields
                for key, value in kwargs.items():
                    if hasattr(prompt, key):
                        setattr(prompt, key, value)
                
                # Update timestamp
                prompt.updated_at = datetime.now().isoformat()
                
                self.save_prompts()
                return prompt
        
        return None
    
    def delete_prompt(self, prompt_id: int) -> bool:
        """Delete a prompt by ID."""
        for i, prompt in enumerate(self.prompts):
            if prompt.id == prompt_id:
                del self.prompts[i]
                self.save_prompts()
                return True
        return False
    
    def get_prompt_by_id(self, prompt_id: int) -> Optional[Prompt]:
        """Get a single prompt by ID."""
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        if not self.prompts:
            return {
                "total_prompts": 0,
                "categories": [],
                "category_counts": {},
                "difficulty_distribution": {},
                "total_tags": 0
            }
        
        category_counts = {}
        difficulty_counts = {}
        all_tags = set()
        
        for prompt in self.prompts:
            # Count categories
            category_counts[prompt.category] = category_counts.get(prompt.category, 0) + 1
            
            # Count difficulties
            difficulty_counts[prompt.difficulty] = difficulty_counts.get(prompt.difficulty, 0) + 1
            
            # Collect tags
            all_tags.update(prompt.tags)
        
        return {
            "total_prompts": len(self.prompts),
            "categories": list(category_counts.keys()),
            "category_counts": category_counts,
            "difficulty_distribution": difficulty_counts,
            "total_tags": len(all_tags),
            "all_tags": sorted(list(all_tags))
        }
    
    def export_to_csv(self, filename: str):
        """Export prompts to CSV file."""
        df = self.get_prompts_dataframe()
        if not df.empty:
            # Convert tags list to string for CSV
            df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else '')
            df.to_csv(filename, index=False)
    
    def import_from_csv(self, filename: str):
        """Import prompts from CSV file."""
        try:
            df = pd.read_csv(filename)
            
            for _, row in df.iterrows():
                # Parse tags back to list
                tags = [tag.strip() for tag in str(row.get('tags', '')).split(',') if tag.strip()]
                
                self.add_prompt(
                    category=row['category'],
                    prompt_text=row['prompt'],
                    description=row['description'],
                    expected_behavior=row['expected_behavior'],
                    difficulty=row.get('difficulty', 'Medium'),
                    tags=tags
                )
                
        except Exception as e:
            print(f"Error importing from CSV: {e}")
    
    def validate_prompts(self) -> List[Dict[str, Any]]:
        """Validate all prompts and return any issues found."""
        issues = []
        
        for prompt in self.prompts:
            prompt_issues = []
            
            # Check required fields
            if not prompt.prompt.strip():
                prompt_issues.append("Empty prompt text")
            
            if not prompt.description.strip():
                prompt_issues.append("Empty description")
            
            if not prompt.category.strip():
                prompt_issues.append("Empty category")
            
            # Check difficulty values
            valid_difficulties = ["Easy", "Medium", "Hard"]
            if prompt.difficulty not in valid_difficulties:
                prompt_issues.append(f"Invalid difficulty: {prompt.difficulty}")
            
            # Check prompt length (should be reasonable)
            if len(prompt.prompt) > 5000:
                prompt_issues.append("Prompt text too long (>5000 characters)")
            
            if len(prompt.prompt) < 10:
                prompt_issues.append("Prompt text too short (<10 characters)")
            
            if prompt_issues:
                issues.append({
                    "prompt_id": prompt.id,
                    "issues": prompt_issues
                })
        
        return issues
