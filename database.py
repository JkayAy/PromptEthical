"""
Database Management Module
Handles SQLite database operations for prompts and execution results.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import os

class DatabaseManager:
    """Manages SQLite database operations for the prompt library."""
    
    def __init__(self, db_path: str = "ethical_ai_prompt_library.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create prompts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prompts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        description TEXT NOT NULL,
                        expected_behavior TEXT,
                        difficulty TEXT DEFAULT 'Medium',
                        tags TEXT,  -- JSON array of tags
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Create execution_results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS execution_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt_id INTEGER NOT NULL,
                        category TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        response TEXT,
                        status TEXT NOT NULL,  -- 'success', 'error', 'timeout'
                        execution_time REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        pass_fail_status TEXT,  -- 'pass', 'fail', 'unclear'
                        usage TEXT,  -- JSON object with token usage
                        error_message TEXT,
                        FOREIGN KEY (prompt_id) REFERENCES prompts (id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_results_timestamp 
                    ON execution_results (timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_results_model 
                    ON execution_results (model_name)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_results_category 
                    ON execution_results (category)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_results_status 
                    ON execution_results (status)
                """)
                
                conn.commit()
                
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def log_execution_result(self, prompt_id: int, category: str, model_name: str,
                           response: str, status: str, execution_time: float,
                           timestamp: str, pass_fail_status: Optional[str] = None,
                           usage: Optional[Dict[str, Any]] = None,
                           error_message: Optional[str] = None) -> int:
        """
        Log execution result to database.
        
        Returns:
            ID of the inserted record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                usage_json = json.dumps(usage) if usage else None
                
                cursor.execute("""
                    INSERT INTO execution_results 
                    (prompt_id, category, model_name, response, status, execution_time, 
                     timestamp, pass_fail_status, usage, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prompt_id, category, model_name, response, status, execution_time,
                    timestamp, pass_fail_status, usage_json, error_message
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                return result_id or -1
                
        except Exception as e:
            print(f"Error logging execution result: {e}")
            return -1
    
    def get_execution_results(self, days: Optional[int] = None, 
                            model_name: Optional[str] = None,
                            category: Optional[str] = None,
                            status: Optional[str] = None) -> pd.DataFrame:
        """
        Get execution results with optional filters.
        
        Args:
            days: Number of days to look back (None for all)
            model_name: Filter by model name
            category: Filter by category
            status: Filter by status
            
        Returns:
            DataFrame with execution results
        """
        try:
            query = "SELECT * FROM execution_results WHERE 1=1"
            params = []
            
            # Date filter
            if days is not None:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                query += " AND timestamp >= ?"
                params.append(cutoff_date)
            
            # Model filter
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            # Category filter
            if category:
                query += " AND category = ?"
                params.append(category)
            
            # Status filter
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse usage JSON if present
                if 'usage' in df.columns:
                    df['usage_parsed'] = df['usage'].apply(
                        lambda x: json.loads(x) if x else {}
                    )
                
                return df
                
        except Exception as e:
            print(f"Error getting execution results: {e}")
            return pd.DataFrame()
    
    def get_execution_history(self, prompt_id: Optional[int] = None,
                            model_name: Optional[str] = None,
                            category: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history with optional filters."""
        try:
            query = """
                SELECT er.*, p.prompt, p.description 
                FROM execution_results er
                LEFT JOIN prompts p ON er.prompt_id = p.id
                WHERE 1=1
            """
            params = []
            
            if prompt_id is not None:
                query += " AND er.prompt_id = ?"
                params.append(prompt_id)
            
            if model_name:
                query += " AND er.model_name = ?"
                params.append(model_name)
            
            if category:
                query += " AND er.category = ?"
                params.append(category)
            
            query += " ORDER BY er.timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse usage JSON
                    if result.get('usage'):
                        try:
                            result['usage_parsed'] = json.loads(result['usage'])
                        except:
                            result['usage_parsed'] = {}
                    results.append(result)
                
                return results
                
        except Exception as e:
            print(f"Error getting execution history: {e}")
            return []
    
    def get_model_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get statistics about model performance."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Overall statistics
                overall_query = """
                    SELECT 
                        model_name,
                        COUNT(*) as total_executions,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_executions,
                        AVG(execution_time) as avg_execution_time,
                        SUM(CASE WHEN pass_fail_status = 'pass' THEN 1 ELSE 0 END) as passes,
                        SUM(CASE WHEN pass_fail_status = 'fail' THEN 1 ELSE 0 END) as fails
                    FROM execution_results 
                    WHERE timestamp >= ?
                    GROUP BY model_name
                """
                
                df = pd.read_sql_query(overall_query, conn, params=[cutoff_date])
                
                # Calculate success and pass rates
                df['success_rate'] = (df['successful_executions'] / df['total_executions']) * 100
                df['pass_rate'] = (df['passes'] / (df['passes'] + df['fails'])) * 100
                
                # Category-wise statistics
                category_query = """
                    SELECT 
                        model_name,
                        category,
                        COUNT(*) as executions,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes
                    FROM execution_results 
                    WHERE timestamp >= ?
                    GROUP BY model_name, category
                """
                
                category_df = pd.read_sql_query(category_query, conn, params=[cutoff_date])
                category_df['success_rate'] = (category_df['successes'] / category_df['executions']) * 100
                
                return {
                    "overall_stats": df.to_dict('records'),
                    "category_stats": category_df.to_dict('records'),
                    "date_range": f"Last {days} days"
                }
                
        except Exception as e:
            print(f"Error getting model statistics: {e}")
            return {"overall_stats": [], "category_stats": [], "error": str(e)}
    
    def get_category_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics by category."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        category,
                        COUNT(*) as total_executions,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_executions,
                        AVG(execution_time) as avg_execution_time,
                        COUNT(DISTINCT model_name) as models_tested,
                        SUM(CASE WHEN pass_fail_status = 'pass' THEN 1 ELSE 0 END) as passes,
                        SUM(CASE WHEN pass_fail_status = 'fail' THEN 1 ELSE 0 END) as fails,
                        SUM(CASE WHEN pass_fail_status = 'unclear' THEN 1 ELSE 0 END) as unclear
                    FROM execution_results 
                    WHERE timestamp >= ?
                    GROUP BY category
                    ORDER BY total_executions DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[cutoff_date])
                
                # Calculate rates
                df['success_rate'] = (df['successful_executions'] / df['total_executions']) * 100
                df['pass_rate'] = (df['passes'] / (df['passes'] + df['fails'])) * 100
                
                return {
                    "category_stats": df.to_dict('records'),
                    "date_range": f"Last {days} days"
                }
                
        except Exception as e:
            print(f"Error getting category performance: {e}")
            return {"category_stats": [], "error": str(e)}
    
    def cleanup_old_results(self, days: int = 90) -> int:
        """Delete execution results older than specified days."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM execution_results WHERE timestamp < ?",
                    (cutoff_date,)
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                return deleted_count
                
        except Exception as e:
            print(f"Error cleaning up old results: {e}")
            return 0
    
    def export_execution_results(self, filename: str, days: Optional[int] = None):
        """Export execution results to CSV file."""
        try:
            df = self.get_execution_results(days=days)
            if not df.empty:
                df.to_csv(filename, index=False)
                return len(df)
            return 0
        except Exception as e:
            print(f"Error exporting execution results: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get general database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records in each table
                cursor.execute("SELECT COUNT(*) FROM prompts")
                prompt_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM execution_results")
                result_count = cursor.fetchone()[0]
                
                # Get date range of execution results
                cursor.execute("""
                    SELECT MIN(timestamp), MAX(timestamp) 
                    FROM execution_results
                """)
                date_range = cursor.fetchone()
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    "prompt_count": prompt_count,
                    "execution_result_count": result_count,
                    "earliest_result": date_range[0] if date_range[0] else None,
                    "latest_result": date_range[1] if date_range[1] else None,
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            print(f"Error backing up database: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            import shutil
            shutil.copy2(backup_path, self.db_path)
            return True
        except Exception as e:
            print(f"Error restoring database: {e}")
            return False
