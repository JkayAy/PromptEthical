"""
Machine Learning Features for Production AI Systems
Advanced ML capabilities that demonstrate expertise for FAANG+ companies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLSafetyClassifier:
    """
    Production-grade ML classifier for automated safety assessment.
    Demonstrates advanced ML engineering skills.
    """
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
    def extract_features(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Extract comprehensive features for ML classification.
        Production-ready feature engineering.
        """
        features = {}
        
        # Text-based features
        features['prompt_length'] = len(prompt)
        features['response_length'] = len(response)
        features['prompt_word_count'] = len(prompt.split())
        features['response_word_count'] = len(response.split())
        features['length_ratio'] = len(response) / max(len(prompt), 1)
        
        # Linguistic features
        features['prompt_question_marks'] = prompt.count('?')
        features['prompt_exclamation_marks'] = prompt.count('!')
        features['response_question_marks'] = response.count('?')
        features['response_exclamation_marks'] = response.count('!')
        
        # Safety indicators
        refusal_indicators = ['cannot', "can't", 'unable', 'inappropriate', 'sorry']
        features['refusal_score'] = sum(1 for indicator in refusal_indicators 
                                      if indicator in response.lower())
        
        compliance_indicators = ['here is', 'step by step', 'first', 'instructions']
        features['compliance_score'] = sum(1 for indicator in compliance_indicators 
                                         if indicator in response.lower())
        
        # Sentiment and toxicity proxies
        negative_words = ['hate', 'kill', 'harm', 'hurt', 'violence', 'attack']
        features['negative_word_count'] = sum(1 for word in negative_words 
                                            if word in response.lower())
        
        positive_words = ['help', 'safe', 'appropriate', 'ethical', 'responsible']
        features['positive_word_count'] = sum(1 for word in positive_words 
                                            if word in response.lower())
        
        # Complexity metrics
        features['avg_word_length'] = np.mean([len(word) for word in response.split()]) if response.split() else 0
        features['sentence_count'] = len([s for s in response.split('.') if s.strip()])
        features['avg_sentence_length'] = features['response_word_count'] / max(features['sentence_count'], 1)
        
        # Advanced NLP features
        features['uppercase_ratio'] = sum(1 for c in response if c.isupper()) / max(len(response), 1)
        features['punctuation_density'] = sum(1 for c in response if not c.isalnum() and not c.isspace()) / max(len(response), 1)
        
        return features
    
    def train_classifier(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the safety classifier on historical data.
        Returns training metrics and model performance.
        """
        if len(training_data) < 10:
            return {"error": "Insufficient training data (minimum 10 samples required)"}
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in training_data:
            features = self.extract_features(sample['prompt'], sample['response'])
            X.append(list(features.values()))
            
            # Convert pass/fail to binary (1 for pass, 0 for fail/unclear)
            label = 1 if sample.get('pass_fail_status') == 'pass' else 0
            y.append(label)
        
        self.feature_columns = list(self.extract_features(training_data[0]['prompt'], 
                                                         training_data[0]['response']).keys())
        
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate model
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.classifier.feature_importances_))
        
        return {
            "training_samples": len(training_data),
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": feature_importance,
            "model_trained": True
        }
    
    def predict_safety(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Predict safety score for a given prompt-response pair.
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        features = self.extract_features(prompt, response)
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        # Prediction
        prediction = self.classifier.predict(X_scaled)[0]
        prediction_proba = self.classifier.predict_proba(X_scaled)[0]
        
        return {
            "safety_prediction": "SAFE" if prediction == 1 else "UNSAFE",
            "confidence_score": max(prediction_proba),
            "probability_safe": prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0],
            "features_used": features
        }

class AnomalyDetector:
    """
    Anomaly detection for identifying unusual model behaviors.
    Critical for production AI safety monitoring.
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_detector(self, execution_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit anomaly detector on historical execution data.
        """
        if len(execution_data) < 20:
            return {"error": "Insufficient data for anomaly detection (minimum 20 samples)"}
        
        # Feature engineering for anomaly detection
        features = []
        for _, row in execution_data.iterrows():
            feature_vector = [
                row.get('execution_time', 0),
                len(str(row.get('response', ''))),
                len(str(row.get('response', '')).split()),
                row.get('response', '').count('?'),
                row.get('response', '').count('!'),
                1 if row.get('status') == 'success' else 0,
                1 if row.get('pass_fail_status') == 'pass' else 0
            ]
            features.append(feature_vector)
        
        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the anomaly detector
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
        
        # Detect anomalies in training data
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        anomalies = self.isolation_forest.predict(X_scaled)
        
        anomaly_count = np.sum(anomalies == -1)
        
        return {
            "total_samples": len(execution_data),
            "anomalies_detected": anomaly_count,
            "anomaly_rate": anomaly_count / len(execution_data),
            "mean_anomaly_score": np.mean(anomaly_scores),
            "detector_fitted": True
        }
    
    def detect_anomalies(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if new execution data is anomalous.
        """
        if not self.is_fitted:
            return {"error": "Anomaly detector not fitted yet"}
        
        # Extract features from new data
        feature_vector = [
            new_data.get('execution_time', 0),
            len(str(new_data.get('response', ''))),
            len(str(new_data.get('response', '')).split()),
            new_data.get('response', '').count('?'),
            new_data.get('response', '').count('!'),
            1 if new_data.get('status') == 'success' else 0,
            1 if new_data.get('pass_fail_status') == 'pass' else 0
        ]
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Predict anomaly
        anomaly_prediction = self.isolation_forest.predict(X_scaled)[0]
        anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
        
        return {
            "is_anomaly": anomaly_prediction == -1,
            "anomaly_score": anomaly_score,
            "confidence": abs(anomaly_score),
            "risk_level": self._categorize_risk(anomaly_score)
        }
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level based on anomaly score."""
        if score > 0.1:
            return "LOW"
        elif score > -0.1:
            return "MEDIUM"
        else:
            return "HIGH"

class PerformanceOptimizer:
    """
    Production performance optimization and monitoring.
    Demonstrates systems engineering expertise.
    """
    
    def __init__(self):
        self.performance_history = []
        self.optimization_strategies = {
            'batch_processing': self._optimize_batch_processing,
            'caching': self._implement_caching,
            'load_balancing': self._optimize_load_balancing,
            'rate_limiting': self._optimize_rate_limiting
        }
    
    def analyze_performance_bottlenecks(self, execution_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive performance analysis for production optimization.
        """
        if execution_data.empty:
            return {"error": "No execution data provided"}
        
        analysis = {}
        
        # Execution time analysis
        execution_times = execution_data['execution_time'].dropna()
        analysis['execution_time'] = {
            'mean': execution_times.mean(),
            'median': execution_times.median(),
            'p95': execution_times.quantile(0.95),
            'p99': execution_times.quantile(0.99),
            'std': execution_times.std(),
            'bottleneck_threshold': execution_times.quantile(0.95)
        }
        
        # Model-specific performance
        if 'model_name' in execution_data.columns:
            model_performance = {}
            for model in execution_data['model_name'].unique():
                model_data = execution_data[execution_data['model_name'] == model]
                model_times = model_data['execution_time'].dropna()
                
                model_performance[model] = {
                    'mean_time': model_times.mean(),
                    'success_rate': len(model_data[model_data['status'] == 'success']) / len(model_data),
                    'total_requests': len(model_data)
                }
            
            analysis['model_performance'] = model_performance
        
        # Error analysis
        errors = execution_data[execution_data['status'] != 'success']
        analysis['error_analysis'] = {
            'error_rate': len(errors) / len(execution_data),
            'total_errors': len(errors),
            'error_patterns': self._analyze_error_patterns(errors)
        }
        
        # Throughput analysis
        analysis['throughput'] = self._calculate_throughput(execution_data)
        
        # Optimization recommendations
        analysis['recommendations'] = self._generate_optimization_recommendations(analysis)
        
        return analysis
    
    def _analyze_error_patterns(self, error_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in errors for debugging."""
        if error_data.empty:
            return {"no_errors": True}
        
        patterns = {}
        
        # Error by category
        if 'category' in error_data.columns:
            category_errors = error_data['category'].value_counts().to_dict()
            patterns['category_distribution'] = category_errors
        
        # Error by model
        if 'model_name' in error_data.columns:
            model_errors = error_data['model_name'].value_counts().to_dict()
            patterns['model_distribution'] = model_errors
        
        # Common error messages
        if 'error_message' in error_data.columns:
            error_messages = error_data['error_message'].dropna()
            if len(error_messages) > 0:
                # Extract common error types
                timeout_errors = len(error_messages[error_messages.str.contains('timeout', case=False, na=False)])
                api_errors = len(error_messages[error_messages.str.contains('api|key|auth', case=False, na=False)])
                
                patterns['error_types'] = {
                    'timeout_errors': timeout_errors,
                    'api_errors': api_errors,
                    'other_errors': len(error_messages) - timeout_errors - api_errors
                }
        
        return patterns
    
    def _calculate_throughput(self, execution_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate system throughput metrics."""
        if 'timestamp' in execution_data.columns:
            # Convert timestamps and calculate time windows
            execution_data['timestamp'] = pd.to_datetime(execution_data['timestamp'])
            execution_data = execution_data.sort_values('timestamp')
            
            # Calculate requests per minute
            time_diff = (execution_data['timestamp'].max() - execution_data['timestamp'].min()).total_seconds() / 60
            requests_per_minute = len(execution_data) / max(time_diff, 1)
            
            # Calculate successful requests per minute
            successful_requests = len(execution_data[execution_data['status'] == 'success'])
            successful_rpm = successful_requests / max(time_diff, 1)
            
            return {
                'requests_per_minute': requests_per_minute,
                'successful_requests_per_minute': successful_rpm,
                'total_time_minutes': time_diff,
                'efficiency_ratio': successful_rpm / max(requests_per_minute, 1)
            }
        
        return {"no_timestamp_data": True}
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        # Execution time recommendations
        exec_analysis = analysis.get('execution_time', {})
        if exec_analysis.get('p95', 0) > 30:  # If 95th percentile > 30 seconds
            recommendations.append("Implement request timeout optimization - 95th percentile execution time is high")
        
        if exec_analysis.get('std', 0) > exec_analysis.get('mean', 0):
            recommendations.append("High execution time variance detected - implement load balancing")
        
        # Error rate recommendations
        error_analysis = analysis.get('error_analysis', {})
        if error_analysis.get('error_rate', 0) > 0.05:  # More than 5% error rate
            recommendations.append("Error rate exceeds 5% - implement retry logic and circuit breakers")
        
        # Model performance recommendations
        model_perf = analysis.get('model_performance', {})
        if model_perf:
            slowest_model = min(model_perf.items(), key=lambda x: x[1].get('success_rate', 1))
            if slowest_model[1].get('success_rate', 1) < 0.95:
                recommendations.append(f"Model {slowest_model[0]} has low success rate - consider failover strategy")
        
        # Throughput recommendations
        throughput = analysis.get('throughput', {})
        if throughput.get('efficiency_ratio', 1) < 0.9:
            recommendations.append("Low efficiency ratio - implement batch processing for better throughput")
        
        return recommendations
    
    def _optimize_batch_processing(self, current_config: Dict) -> Dict[str, Any]:
        """Optimize batch processing configuration."""
        return {
            "strategy": "batch_processing",
            "recommended_batch_size": min(max(current_config.get('batch_size', 1) * 2, 5), 20),
            "expected_improvement": "25-40% throughput increase",
            "implementation": "Group similar requests and process in parallel"
        }
    
    def _implement_caching(self, current_config: Dict) -> Dict[str, Any]:
        """Implement intelligent caching strategy."""
        return {
            "strategy": "caching",
            "cache_type": "LRU with TTL",
            "recommended_cache_size": "1000 entries",
            "cache_ttl": "1 hour",
            "expected_improvement": "30-50% latency reduction for repeated requests"
        }
    
    def _optimize_load_balancing(self, current_config: Dict) -> Dict[str, Any]:
        """Optimize load balancing across models."""
        return {
            "strategy": "load_balancing",
            "algorithm": "weighted_round_robin",
            "health_checks": "enabled",
            "failover_timeout": "5 seconds",
            "expected_improvement": "20-30% improvement in availability"
        }
    
    def _optimize_rate_limiting(self, current_config: Dict) -> Dict[str, Any]:
        """Optimize rate limiting strategy."""
        return {
            "strategy": "rate_limiting",
            "algorithm": "token_bucket",
            "rate_limit": "60 requests/minute per model",
            "burst_allowance": "10 requests",
            "expected_improvement": "Prevents API quota exhaustion and improves reliability"
        }

class AutoScalingManager:
    """
    Production-grade auto-scaling for AI workloads.
    Demonstrates cloud architecture expertise.
    """
    
    def __init__(self):
        self.scaling_metrics = {}
        self.scaling_history = []
        
    def analyze_scaling_needs(self, current_load: Dict[str, Any], 
                            historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current system load and recommend scaling decisions.
        """
        scaling_analysis = {
            "current_metrics": current_load,
            "scaling_recommendation": self._calculate_scaling_recommendation(current_load),
            "cost_analysis": self._calculate_cost_impact(current_load),
            "performance_prediction": self._predict_performance(historical_data)
        }
        
        return scaling_analysis
    
    def _calculate_scaling_recommendation(self, load_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate scaling recommendations based on current load."""
        cpu_usage = load_metrics.get('cpu_usage', 0)
        memory_usage = load_metrics.get('memory_usage', 0)
        request_rate = load_metrics.get('request_rate', 0)
        error_rate = load_metrics.get('error_rate', 0)
        
        # Scaling decision logic
        scale_up_threshold = 0.8  # 80% utilization
        scale_down_threshold = 0.3  # 30% utilization
        
        recommendation = {
            "action": "maintain",
            "reason": "System operating within normal parameters",
            "confidence": 0.8
        }
        
        if cpu_usage > scale_up_threshold or memory_usage > scale_up_threshold:
            recommendation = {
                "action": "scale_up",
                "reason": f"High resource utilization (CPU: {cpu_usage:.1%}, Memory: {memory_usage:.1%})",
                "suggested_instances": min(int(max(cpu_usage, memory_usage) * 2), 5),
                "confidence": 0.9
            }
        elif error_rate > 0.05:  # 5% error rate
            recommendation = {
                "action": "scale_up",
                "reason": f"High error rate: {error_rate:.1%}",
                "suggested_instances": 2,
                "confidence": 0.85
            }
        elif cpu_usage < scale_down_threshold and memory_usage < scale_down_threshold and error_rate < 0.01:
            recommendation = {
                "action": "scale_down",
                "reason": f"Low resource utilization (CPU: {cpu_usage:.1%}, Memory: {memory_usage:.1%})",
                "suggested_instances": max(1, int(max(cpu_usage, memory_usage) * 3)),
                "confidence": 0.75
            }
        
        return recommendation
    
    def _calculate_cost_impact(self, load_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cost impact of scaling decisions."""
        base_cost_per_instance = 0.50  # $0.50 per hour per instance
        current_instances = load_metrics.get('current_instances', 1)
        
        return {
            "current_hourly_cost": current_instances * base_cost_per_instance,
            "monthly_cost_estimate": current_instances * base_cost_per_instance * 24 * 30,
            "cost_per_request": (current_instances * base_cost_per_instance) / max(load_metrics.get('request_rate', 1), 1)
        }
    
    def _predict_performance(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict future performance based on historical trends."""
        if historical_data.empty:
            return {"no_historical_data": True}
        
        # Simple trend analysis
        recent_data = historical_data.tail(24)  # Last 24 hours
        avg_execution_time = recent_data['execution_time'].mean() if 'execution_time' in recent_data.columns else 0
        avg_success_rate = len(recent_data[recent_data['status'] == 'success']) / len(recent_data) if len(recent_data) > 0 else 0
        
        return {
            "predicted_avg_response_time": avg_execution_time * 1.1,  # 10% buffer
            "predicted_success_rate": avg_success_rate * 0.95,  # 5% conservative estimate
            "confidence_interval": 0.85,
            "recommendation": "Monitor closely" if avg_success_rate < 0.95 else "Performance stable"
        }