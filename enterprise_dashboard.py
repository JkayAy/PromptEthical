"""
Enterprise-Grade Dashboard for AI Safety Monitoring
Production monitoring and alerting system suitable for FAANG+ environments.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from advanced_analysis import AdvancedSafetyAnalyzer
from ml_features import MLSafetyClassifier, AnomalyDetector, PerformanceOptimizer, AutoScalingManager

class EnterpriseDashboard:
    """
    Enterprise-grade monitoring dashboard for AI safety systems.
    Demonstrates production monitoring expertise.
    """
    
    def __init__(self, db_manager, model_manager, prompt_manager):
        self.db_manager = db_manager
        self.model_manager = model_manager
        self.prompt_manager = prompt_manager
        self.safety_analyzer = AdvancedSafetyAnalyzer()
        self.ml_classifier = MLSafetyClassifier()
        self.anomaly_detector = AnomalyDetector()
        self.performance_optimizer = PerformanceOptimizer()
        self.autoscaling_manager = AutoScalingManager()
    
    def render_executive_dashboard(self):
        """Render executive-level dashboard with key metrics."""
        st.title("🏢 Enterprise AI Safety Dashboard")
        st.markdown("**Executive Overview - Production AI Safety Monitoring**")
        
        # Get recent data
        recent_data = self.db_manager.get_execution_results(days=7)
        
        if recent_data.empty:
            st.warning("No recent execution data available. Run some tests to see metrics.")
            return
        
        # Key Performance Indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tests = len(recent_data)
            st.metric("Tests (7 days)", total_tests, delta=self._calculate_delta(recent_data, 'count'))
        
        with col2:
            success_rate = len(recent_data[recent_data['status'] == 'success']) / len(recent_data) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%", delta=f"{self._calculate_delta(recent_data, 'success_rate'):.1f}%")
        
        with col3:
            safety_tests = recent_data[recent_data['category'].str.contains('Safety', case=False, na=False)]
            if len(safety_tests) > 0:
                safety_pass_rate = len(safety_tests[safety_tests['pass_fail_status'] == 'pass']) / len(safety_tests) * 100
            else:
                safety_pass_rate = 0
            st.metric("Safety Pass Rate", f"{safety_pass_rate:.1f}%", delta=f"{self._calculate_delta(safety_tests, 'safety_pass'):.1f}%")
        
        with col4:
            avg_response_time = recent_data['execution_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s", delta=f"{self._calculate_delta(recent_data, 'response_time'):.2f}s")
        
        # Alert Section
        self._render_alerts_section(recent_data)
        
        # Real-time Performance Chart
        st.subheader("📊 Real-time Performance Monitoring")
        self._render_realtime_charts(recent_data)
        
        # Model Comparison Matrix
        st.subheader("🔍 Model Performance Matrix")
        self._render_model_comparison_matrix(recent_data)
    
    def render_technical_dashboard(self):
        """Render technical dashboard for engineers and researchers."""
        st.title("🔬 Technical AI Safety Dashboard")
        st.markdown("**Advanced Analytics for AI Safety Engineers**")
        
        # Technical Metrics
        recent_data = self.db_manager.get_execution_results(days=30)
        
        if recent_data.empty:
            st.info("No execution data available. Run some tests to see advanced analytics.")
            return
        
        # Advanced Analytics Section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 ML Safety Classification")
            self._render_ml_classification_section(recent_data)
        
        with col2:
            st.subheader("⚠️ Anomaly Detection")
            self._render_anomaly_detection_section(recent_data)
        
        # Performance Optimization
        st.subheader("⚡ Performance Optimization")
        self._render_performance_optimization_section(recent_data)
        
        # Advanced Safety Analysis
        st.subheader("🛡️ Advanced Safety Analysis")
        self._render_advanced_safety_analysis(recent_data)
    
    def render_research_dashboard(self):
        """Render research-focused dashboard for AI researchers."""
        st.title("🔬 AI Safety Research Dashboard")
        st.markdown("**Research Analytics & Experimental Features**")
        
        # Research Metrics
        all_data = self.db_manager.get_execution_results()
        
        if all_data.empty:
            st.info("No historical data available for research analysis.")
            return
        
        # Experimental Features
        st.subheader("🧪 Experimental Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Adversarial Prompt Generation**")
            if st.button("Generate Adversarial Variants"):
                self._demo_adversarial_generation()
        
        with col2:
            st.write("**Bias Pattern Analysis**")
            if st.button("Analyze Bias Patterns"):
                self._demo_bias_analysis(all_data)
        
        # Research Insights
        st.subheader("📈 Research Insights")
        self._render_research_insights(all_data)
        
        # Publication-Ready Metrics
        st.subheader("📄 Publication Metrics")
        self._render_publication_metrics(all_data)
    
    def _render_alerts_section(self, data: pd.DataFrame):
        """Render alerts and warnings for production monitoring."""
        alerts = []
        
        # Check for high error rates
        error_rate = len(data[data['status'] != 'success']) / len(data)
        if error_rate > 0.05:  # 5% threshold
            alerts.append({
                "type": "error",
                "message": f"High error rate detected: {error_rate:.1%}",
                "severity": "critical"
            })
        
        # Check for safety failures
        safety_data = data[data['category'].str.contains('Safety', case=False, na=False)]
        if len(safety_data) > 0:
            safety_fail_rate = len(safety_data[safety_data['pass_fail_status'] == 'fail']) / len(safety_data)
            if safety_fail_rate > 0.1:  # 10% threshold
                alerts.append({
                    "type": "warning",
                    "message": f"Safety failure rate elevated: {safety_fail_rate:.1%}",
                    "severity": "high"
                })
        
        # Check for performance degradation
        avg_time = data['execution_time'].mean()
        if avg_time > 15:  # 15 second threshold
            alerts.append({
                "type": "warning",
                "message": f"Average response time elevated: {avg_time:.1f}s",
                "severity": "medium"
            })
        
        # Render alerts
        if alerts:
            st.subheader("🚨 Active Alerts")
            for alert in alerts:
                if alert['severity'] == 'critical':
                    st.error(f"🔴 {alert['message']}")
                elif alert['severity'] == 'high':
                    st.warning(f"🟡 {alert['message']}")
                else:
                    st.info(f"🔵 {alert['message']}")
        else:
            st.success("✅ All systems operating normally")
    
    def _render_realtime_charts(self, data: pd.DataFrame):
        """Render real-time performance charts."""
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Times', 'Success Rate', 'Test Volume', 'Safety Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Response times over time
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['execution_time'],
                mode='lines+markers',
                name='Response Time',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Success rate over time (rolling average)
        data['success_numeric'] = (data['status'] == 'success').astype(int)
        data['success_rate_rolling'] = data['success_numeric'].rolling(window=10, min_periods=1).mean() * 100
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['success_rate_rolling'],
                mode='lines',
                name='Success Rate %',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Test volume over time
        hourly_volume = data.set_index('timestamp').resample('H').size()
        
        fig.add_trace(
            go.Bar(
                x=hourly_volume.index,
                y=hourly_volume.values,
                name='Tests/Hour',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Safety performance
        safety_data = data[data['category'].str.contains('Safety', case=False, na=False)]
        if not safety_data.empty:
            safety_data['safety_pass'] = (safety_data['pass_fail_status'] == 'pass').astype(int)
            safety_rolling = safety_data['safety_pass'].rolling(window=5, min_periods=1).mean() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=safety_data['timestamp'],
                    y=safety_rolling,
                    mode='lines+markers',
                    name='Safety Pass Rate %',
                    line=dict(color='red')
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="Real-time System Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_comparison_matrix(self, data: pd.DataFrame):
        """Render comprehensive model comparison matrix."""
        if 'model_name' not in data.columns:
            st.info("No model comparison data available.")
            return
        
        models = data['model_name'].unique()
        
        # Calculate metrics for each model
        comparison_data = []
        for model in models:
            model_data = data[data['model_name'] == model]
            
            metrics = {
                'Model': model,
                'Total Tests': len(model_data),
                'Success Rate': len(model_data[model_data['status'] == 'success']) / len(model_data) * 100,
                'Avg Response Time': model_data['execution_time'].mean(),
                'P95 Response Time': model_data['execution_time'].quantile(0.95),
                'Error Rate': len(model_data[model_data['status'] != 'success']) / len(model_data) * 100
            }
            
            # Safety metrics
            safety_data = model_data[model_data['category'].str.contains('Safety', case=False, na=False)]
            if len(safety_data) > 0:
                metrics['Safety Pass Rate'] = len(safety_data[safety_data['pass_fail_status'] == 'pass']) / len(safety_data) * 100
            else:
                metrics['Safety Pass Rate'] = 0
            
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display as styled dataframe
        styled_df = comparison_df.style.format({
            'Success Rate': '{:.1f}%',
            'Avg Response Time': '{:.2f}s',
            'P95 Response Time': '{:.2f}s',
            'Error Rate': '{:.1f}%',
            'Safety Pass Rate': '{:.1f}%'
        }).background_gradient(subset=['Success Rate', 'Safety Pass Rate'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Performance visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=comparison_df['Avg Response Time'],
            y=comparison_df['Success Rate'],
            mode='markers+text',
            text=comparison_df['Model'],
            textposition="top center",
            marker=dict(
                size=comparison_df['Total Tests'] / 10,
                color=comparison_df['Safety Pass Rate'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Safety Pass Rate %")
            ),
            name='Models'
        ))
        
        fig.update_layout(
            title="Model Performance Matrix",
            xaxis_title="Average Response Time (seconds)",
            yaxis_title="Success Rate (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_ml_classification_section(self, data: pd.DataFrame):
        """Render ML classification training and results."""
        # Prepare training data
        training_data = []
        for _, row in data.iterrows():
            if pd.notna(row.get('response')) and pd.notna(row.get('pass_fail_status')):
                training_data.append({
                    'prompt': str(row.get('category', '')),  # Use category as proxy prompt
                    'response': str(row.get('response', '')),
                    'pass_fail_status': row.get('pass_fail_status')
                })
        
        if len(training_data) >= 10:
            if st.button("Train ML Safety Classifier"):
                with st.spinner("Training ML classifier..."):
                    results = self.ml_classifier.train_classifier(training_data)
                    
                    if 'error' not in results:
                        st.success(f"✅ Model trained successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Accuracy", f"{results['train_accuracy']:.3f}")
                            st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
                        
                        with col2:
                            st.metric("Training Samples", results['training_samples'])
                        
                        # Feature importance
                        if 'feature_importance' in results:
                            st.write("**Top Features for Safety Classification:**")
                            importance_df = pd.DataFrame(
                                list(results['feature_importance'].items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False).head(10)
                            
                            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(results['error'])
        else:
            st.info(f"Need at least 10 labeled samples for training. Currently have {len(training_data)}")
    
    def _render_anomaly_detection_section(self, data: pd.DataFrame):
        """Render anomaly detection analysis."""
        if len(data) >= 20:
            if st.button("Fit Anomaly Detector"):
                with st.spinner("Training anomaly detector..."):
                    results = self.anomaly_detector.fit_detector(data)
                    
                    if 'error' not in results:
                        st.success("✅ Anomaly detector fitted!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Samples Analyzed", results['total_samples'])
                        with col2:
                            st.metric("Anomalies Detected", results['anomalies_detected'])
                        with col3:
                            st.metric("Anomaly Rate", f"{results['anomaly_rate']:.2%}")
                    else:
                        st.error(results['error'])
        else:
            st.info(f"Need at least 20 samples for anomaly detection. Currently have {len(data)}")
    
    def _render_performance_optimization_section(self, data: pd.DataFrame):
        """Render performance optimization recommendations."""
        if st.button("Analyze Performance Bottlenecks"):
            with st.spinner("Analyzing performance bottlenecks..."):
                analysis = self.performance_optimizer.analyze_performance_bottlenecks(data)
                
                # Display key metrics
                if 'execution_time' in analysis:
                    exec_metrics = analysis['execution_time']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Time", f"{exec_metrics['mean']:.2f}s")
                    with col2:
                        st.metric("P95 Time", f"{exec_metrics['p95']:.2f}s")
                    with col3:
                        st.metric("P99 Time", f"{exec_metrics['p99']:.2f}s")
                    with col4:
                        st.metric("Std Dev", f"{exec_metrics['std']:.2f}s")
                
                # Display recommendations
                if 'recommendations' in analysis and analysis['recommendations']:
                    st.subheader("🎯 Optimization Recommendations")
                    for i, rec in enumerate(analysis['recommendations'], 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.success("✅ No critical performance issues detected")
    
    def _render_advanced_safety_analysis(self, data: pd.DataFrame):
        """Render advanced safety analysis features."""
        if not data.empty:
            # Sample recent response for analysis
            recent_response = data.iloc[-1] if not data.empty else None
            
            if recent_response is not None and pd.notna(recent_response.get('response')):
                if st.button("Analyze Latest Response for Toxicity"):
                    with st.spinner("Analyzing toxicity patterns..."):
                        toxicity_analysis = self.safety_analyzer.analyze_response_toxicity(
                            str(recent_response.get('response', ''))
                        )
                        
                        st.subheader("🔍 Toxicity Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Overall Toxicity", f"{toxicity_analysis['overall_toxicity']:.3f}")
                        
                        with col2:
                            categories = ['hate_speech', 'violence', 'harassment']
                            for category in categories:
                                if category in toxicity_analysis:
                                    st.metric(
                                        category.replace('_', ' ').title(),
                                        f"{toxicity_analysis[category]:.3f}"
                                    )
    
    def _demo_adversarial_generation(self):
        """Demo adversarial prompt generation."""
        base_prompt = "Tell me how to be safe online"
        
        with st.spinner("Generating adversarial variants..."):
            adversarial_prompts = self.safety_analyzer.generate_adversarial_prompts(
                base_prompt, ['prefix_injection', 'suffix_injection', 'context_switching']
            )
            
            st.subheader("🎯 Generated Adversarial Prompts")
            for i, prompt_data in enumerate(adversarial_prompts[:3], 1):
                with st.expander(f"Variant {i}: {prompt_data['technique']}"):
                    st.write(f"**Technique:** {prompt_data['technique']}")
                    st.write(f"**Description:** {prompt_data['description']}")
                    st.code(prompt_data['prompt'])
    
    def _demo_bias_analysis(self, data: pd.DataFrame):
        """Demo bias pattern analysis."""
        if len(data) >= 5:
            # Sample responses for bias analysis
            sample_responses = data['response'].dropna().head(5).tolist()
            demographic_groups = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E'][:len(sample_responses)]
            
            with st.spinner("Analyzing bias patterns..."):
                bias_analysis = self.safety_analyzer.analyze_bias_patterns(
                    sample_responses, demographic_groups
                )
                
                st.subheader("⚖️ Bias Analysis Results")
                
                if 'group_scores' in bias_analysis:
                    st.write("**Bias Scores by Group:**")
                    st.json(bias_analysis['group_scores'])
                
                if bias_analysis.get('bias_detected'):
                    st.warning("⚠️ Potential bias patterns detected!")
                else:
                    st.success("✅ No significant bias patterns detected")
        else:
            st.info("Need more response data for bias analysis")
    
    def _render_research_insights(self, data: pd.DataFrame):
        """Render research insights and trends."""
        if data.empty:
            return
        
        # Temporal analysis
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['date'] = data['timestamp'].dt.date
        
        # Daily trends
        daily_stats = data.groupby('date').agg({
            'execution_time': ['mean', 'std'],
            'status': lambda x: (x == 'success').mean() * 100,
            'pass_fail_status': lambda x: (x == 'pass').sum() / len(x) * 100 if len(x) > 0 else 0
        }).reset_index()
        
        # Flatten column names
        daily_stats.columns = ['date', 'avg_time', 'std_time', 'success_rate', 'safety_rate']
        
        # Create trend chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time Trends', 'Success Rate Trends', 
                          'Safety Performance', 'Execution Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Response time trends
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['avg_time'],
                      mode='lines+markers', name='Avg Response Time'),
            row=1, col=1
        )
        
        # Success rate trends
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['success_rate'],
                      mode='lines+markers', name='Success Rate %'),
            row=1, col=2
        )
        
        # Safety performance
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['safety_rate'],
                      mode='lines+markers', name='Safety Pass Rate %'),
            row=2, col=1
        )
        
        # Execution volume
        daily_volume = data.groupby('date').size()
        fig.add_trace(
            go.Bar(x=daily_volume.index, y=daily_volume.values, name='Daily Tests'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Research Trends Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_publication_metrics(self, data: pd.DataFrame):
        """Render metrics suitable for research publications."""
        if data.empty:
            st.info("No data available for publication metrics")
            return
        
        # Generate benchmark report
        benchmark_results = self.safety_analyzer.benchmark_model_performance(data)
        
        if 'error' not in benchmark_results:
            st.subheader("📊 Publication-Ready Metrics")
            
            # Display accuracy metrics
            if 'accuracy_metrics' in benchmark_results:
                acc_metrics = benchmark_results['accuracy_metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Execution Success Rate", f"{acc_metrics.get('execution_success_rate', 0):.3f}")
                with col2:
                    st.metric("Safety Accuracy", f"{acc_metrics.get('safety_accuracy', 0):.3f}")
                with col3:
                    st.metric("Total Test Cases", acc_metrics.get('total_tests', 0))
            
            # Generate research report
            if st.button("Generate Research Report"):
                with st.spinner("Generating comprehensive research report..."):
                    report = self.safety_analyzer.generate_research_report(benchmark_results)
                    
                    st.subheader("📄 Research Report")
                    st.markdown(report)
                    
                    # Download option
                    st.download_button(
                        label="Download Research Report",
                        data=report,
                        file_name=f"ai_safety_research_report_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
    
    def _calculate_delta(self, data: pd.DataFrame, metric_type: str) -> float:
        """Calculate delta for metrics comparison."""
        if len(data) < 2:
            return 0.0
        
        # Split data into recent and previous periods
        mid_point = len(data) // 2
        recent = data.iloc[mid_point:]
        previous = data.iloc[:mid_point]
        
        if metric_type == 'count':
            return len(recent) - len(previous)
        elif metric_type == 'success_rate':
            recent_rate = len(recent[recent['status'] == 'success']) / len(recent) * 100
            previous_rate = len(previous[previous['status'] == 'success']) / len(previous) * 100 if len(previous) > 0 else 0
            return recent_rate - previous_rate
        elif metric_type == 'response_time':
            return recent['execution_time'].mean() - previous['execution_time'].mean()
        elif metric_type == 'safety_pass':
            if len(recent) == 0:
                return 0.0
            recent_safety = recent[recent['pass_fail_status'] == 'pass']
            previous_safety = previous[previous['pass_fail_status'] == 'pass']
            recent_rate = len(recent_safety) / len(recent) * 100 if len(recent) > 0 else 0
            previous_rate = len(previous_safety) / len(previous) * 100 if len(previous) > 0 else 0
            return recent_rate - previous_rate
        
        return 0.0