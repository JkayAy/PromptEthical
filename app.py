"""
Ethical AI Prompt Library - Main Streamlit Application
A production-ready web app for testing LLM safety and robustness.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional

from database import DatabaseManager
from prompts import PromptManager
from runner import PromptRunner
from models import ModelManager
from utils import format_timestamp, export_to_csv, export_to_json

# Page configuration
st.set_page_config(
    page_title="Ethical AI Prompt Library",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers
@st.cache_resource
def init_managers():
    """Initialize all manager instances."""
    db_manager = DatabaseManager()
    prompt_manager = PromptManager()
    model_manager = ModelManager()
    runner = PromptRunner(db_manager, model_manager)
    return db_manager, prompt_manager, model_manager, runner

db_manager, prompt_manager, model_manager, runner = init_managers()

# Sidebar
st.sidebar.title("🛡️ Ethical AI Prompt Library")
st.sidebar.markdown("Testing LLM safety and robustness")

# Main navigation
tab1, tab2, tab3, tab4 = st.tabs(["📚 Prompt Library", "🔬 Run Tests", "📊 Results", "📈 Analysis"])

with tab1:
    st.header("Prompt Library")
    st.markdown("Browse and explore prompts organized by safety categories")
    
    # Load prompts
    prompts_df = prompt_manager.get_prompts_dataframe()
    
    if not prompts_df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=prompts_df['category'].unique(),
                default=prompts_df['category'].unique()
            )
        
        with col2:
            search_term = st.text_input("Search prompts", placeholder="Enter keywords...")
        
        with col3:
            difficulty_filter = st.selectbox(
                "Difficulty Level",
                options=["All"] + list(prompts_df['difficulty'].unique()),
                index=0
            )
        
        # Apply filters
        filtered_df = prompts_df[prompts_df['category'].isin(selected_categories)]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['prompt'].astype(str).str.contains(search_term, case=False, na=False) |
                filtered_df['description'].astype(str).str.contains(search_term, case=False, na=False)
            ]
        
        if difficulty_filter != "All":
            filtered_df = filtered_df[filtered_df['difficulty'] == difficulty_filter]
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Prompts", len(filtered_df))
        with col2:
            st.metric("Categories", len(filtered_df['category'].unique()))
        with col3:
            difficulty_scores = filtered_df['difficulty'].map({'Easy': 1, 'Medium': 2, 'Hard': 3})
            avg_difficulty = difficulty_scores.mean() if len(difficulty_scores) > 0 else 0
            st.metric("Avg Difficulty", f"{avg_difficulty:.1f}/3")
        with col4:
            safety_tests = len(filtered_df[filtered_df['category'].astype(str).str.contains('Safety|Jailbreak', case=False)])
            st.metric("Safety Tests", safety_tests)
        
        # Display prompts table
        st.subheader(f"Prompts ({len(filtered_df)} found)")
        
        # Configure display columns
        display_columns = ['id', 'category', 'difficulty', 'description', 'expected_behavior']
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "category": st.column_config.TextColumn("Category", width="medium"),
                "difficulty": st.column_config.TextColumn("Difficulty", width="small"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "expected_behavior": st.column_config.TextColumn("Expected Behavior", width="medium")
            }
        )
        
        # Show detailed prompt view
        if st.checkbox("Show detailed prompt view"):
            selected_prompt_id = st.selectbox(
                "Select prompt to view details",
                options=filtered_df['id'].tolist(),
                format_func=lambda x: f"#{x}: {filtered_df[filtered_df['id']==x]['description'].iloc[0][:50]}..."
            )
            
            if selected_prompt_id:
                prompt_details = filtered_df[filtered_df['id'] == selected_prompt_id].iloc[0]
                
                st.subheader(f"Prompt #{selected_prompt_id}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Category:**", prompt_details['category'])
                    st.write("**Difficulty:**", prompt_details['difficulty'])
                    st.write("**Description:**", prompt_details['description'])
                
                with col2:
                    st.write("**Expected Behavior:**", prompt_details['expected_behavior'])
                    if prompt_details['tags']:
                        st.write("**Tags:**", ", ".join(prompt_details['tags']))
                
                st.write("**Full Prompt:**")
                st.code(prompt_details['prompt'], language="text")
    else:
        st.warning("No prompts found. Please check the prompts.json file.")

with tab2:
    st.header("Run Tests")
    st.markdown("Execute prompts against selected LLM models")
    
    # Load available prompts and models
    prompts_df = prompt_manager.get_prompts_dataframe()
    available_models = model_manager.get_available_models()
    
    if prompts_df.empty:
        st.error("No prompts available. Please check the prompt library.")
        st.stop()
    
    if not available_models:
        st.error("No models configured. Please check your API keys in environment variables.")
        st.stop()
    
    # Test configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Prompts")
        
        # Category selection
        categories = prompts_df['category'].unique()
        selected_categories = st.multiselect(
            "Categories to test",
            options=categories,
            default=categories[:2] if len(categories) >= 2 else categories
        )
        
        # Individual prompt selection
        selected_prompt_ids = []
        if selected_categories:
            category_prompts = prompts_df[prompts_df['category'].isin(selected_categories)]
            
            prompt_selection_mode = st.radio(
                "Prompt selection",
                ["All prompts in selected categories", "Specific prompts"]
            )
            
            if prompt_selection_mode == "Specific prompts":
                selected_prompt_ids = st.multiselect(
                    "Select specific prompts",
                    options=category_prompts['id'].tolist(),
                    format_func=lambda x: f"#{x}: {category_prompts[category_prompts['id']==x]['description'].iloc[0][:50]}..." if len(category_prompts[category_prompts['id']==x]) > 0 else f"#{x}"
                )
            else:
                selected_prompt_ids = category_prompts['id'].tolist()
            
            st.info(f"Selected {len(selected_prompt_ids)} prompts")
    
    with col2:
        st.subheader("Model Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose model",
            options=list(available_models.keys()),
            format_func=lambda x: f"{x} ({available_models[x]['provider']})"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_tokens = st.slider("Max tokens", 50, 2000, 500)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
            timeout = st.slider("Timeout (seconds)", 10, 120, 30)
            
            # Rate limiting
            rate_limit = st.slider("Requests per minute", 1, 60, 10)
            delay_between_requests = 60 / rate_limit
    
    # Execution controls
    st.subheader("Execution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Test Run", type="primary", use_container_width=True):
            if not selected_prompt_ids:
                st.error("Please select at least one prompt to test.")
            else:
                # Initialize session state for tracking
                st.session_state.test_running = True
                st.session_state.test_results = []
                st.session_state.current_prompt_idx = 0
                st.session_state.test_prompts = selected_prompt_ids
                st.session_state.test_model = selected_model
                
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Tests", use_container_width=True):
            if hasattr(st.session_state, 'test_running'):
                st.session_state.test_running = False
                st.success("Test run stopped.")
    
    with col3:
        if st.button("🗑️ Clear Results", use_container_width=True):
            if hasattr(st.session_state, 'test_results'):
                st.session_state.test_results = []
                st.success("Results cleared.")
    
    # Real-time execution display
    if hasattr(st.session_state, 'test_running') and st.session_state.test_running:
        st.subheader("Test Execution Progress")
        
        progress_placeholder = st.empty()
        results_placeholder = st.empty()
        
        # Execute prompts
        total_prompts = len(st.session_state.test_prompts)
        
        for idx, prompt_id in enumerate(st.session_state.test_prompts):
            if not st.session_state.test_running:
                break
                
            # Update progress
            progress = (idx + 1) / total_prompts
            progress_placeholder.progress(progress, f"Testing prompt {idx + 1}/{total_prompts}")
            
            # Get prompt details
            prompt_row = prompts_df[prompts_df['id'] == prompt_id].iloc[0]
            
            # Execute prompt
            with st.spinner(f"Running prompt #{prompt_id}..."):
                try:
                    result = runner.execute_single_prompt(
                        prompt_text=prompt_row['prompt'],
                        model_name=selected_model,
                        prompt_id=prompt_id,
                        category=prompt_row['category'],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout
                    )
                    
                    st.session_state.test_results.append(result)
                    
                    # Display current results
                    results_df = pd.DataFrame(st.session_state.test_results)
                    results_placeholder.dataframe(
                        results_df[['prompt_id', 'category', 'status', 'execution_time', 'timestamp']],
                        use_container_width=True
                    )
                    
                    # Rate limiting delay
                    if idx < total_prompts - 1:  # Don't delay after last prompt
                        time.sleep(delay_between_requests)
                        
                except Exception as e:
                    st.error(f"Error executing prompt #{prompt_id}: {str(e)}")
                    result = {
                        'prompt_id': prompt_id,
                        'category': prompt_row['category'],
                        'status': 'error',
                        'response': f"Error: {str(e)}",
                        'execution_time': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.test_results.append(result)
        
        # Test completion
        st.session_state.test_running = False
        st.success(f"✅ Test run completed! Executed {len(st.session_state.test_results)} prompts.")
    
    # Show recent results
    if hasattr(st.session_state, 'test_results') and st.session_state.test_results:
        st.subheader("Recent Test Results")
        
        results_df = pd.DataFrame(st.session_state.test_results)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Executed", len(results_df))
        with col2:
            successful = len(results_df[results_df['status'] == 'success'])
            st.metric("Successful", successful)
        with col3:
            failed = len(results_df[results_df['status'] == 'error'])
            st.metric("Failed", failed)
        with col4:
            avg_time = results_df['execution_time'].mean() if 'execution_time' in results_df else 0
            st.metric("Avg Time (s)", f"{avg_time:.2f}")
        
        # Results table
        display_cols = ['prompt_id', 'category', 'status', 'execution_time', 'timestamp']
        if 'response' in results_df.columns:
            display_cols.append('response')
        
        st.dataframe(
            results_df[display_cols],
            use_container_width=True,
            column_config={
                "response": st.column_config.TextColumn("Response", width="large"),
                "execution_time": st.column_config.NumberColumn("Time (s)", format="%.2f")
            }
        )

with tab3:
    st.header("Results History")
    st.markdown("View and analyze historical test results")
    
    # Load historical results
    results_df = db_manager.get_execution_results()
    
    if not results_df.empty:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            date_range = st.date_input(
                "Date range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            model_filter = st.multiselect(
                "Models",
                options=results_df['model_name'].unique(),
                default=results_df['model_name'].unique()
            )
        
        with col3:
            category_filter = st.multiselect(
                "Categories",
                options=results_df['category'].unique(),
                default=results_df['category'].unique()
            )
        
        with col4:
            status_filter = st.multiselect(
                "Status",
                options=results_df['status'].unique(),
                default=results_df['status'].unique()
            )
        
        # Apply filters
        filtered_results = results_df[
            (results_df['model_name'].isin(model_filter)) &
            (results_df['category'].isin(category_filter)) &
            (results_df['status'].isin(status_filter))
        ]
        
        # Date filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_results['date'] = pd.to_datetime(filtered_results['timestamp']).dt.date
            filtered_results = filtered_results[
                (filtered_results['date'] >= start_date) &
                (filtered_results['date'] <= end_date)
            ]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Results", len(filtered_results))
        with col2:
            success_rate = len(filtered_results[filtered_results['status'] == 'success']) / len(filtered_results) * 100 if len(filtered_results) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            avg_exec_time = filtered_results['execution_time'].mean() if len(filtered_results) > 0 else 0
            st.metric("Avg Execution Time", f"{avg_exec_time:.2f}s")
        with col4:
            unique_prompts = filtered_results['prompt_id'].nunique() if len(filtered_results) > 0 else 0
            st.metric("Unique Prompts", unique_prompts)
        
        # Results table
        st.subheader(f"Results ({len(filtered_results)} found)")
        
        # Configure columns for display
        display_columns = ['id', 'prompt_id', 'category', 'model_name', 'status', 'execution_time', 'timestamp']
        
        st.dataframe(
            filtered_results[display_columns],
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("Run ID", width="small"),
                "prompt_id": st.column_config.NumberColumn("Prompt ID", width="small"),
                "execution_time": st.column_config.NumberColumn("Time (s)", format="%.2f", width="small"),
                "timestamp": st.column_config.DatetimeColumn("Timestamp", width="medium")
            }
        )
        
        # Detailed view
        if st.checkbox("Show detailed result view"):
            selected_result_id = st.selectbox(
                "Select result to view details",
                options=filtered_results['id'].tolist(),
                format_func=lambda x: f"Run #{x}: {filtered_results[filtered_results['id']==x]['category'].iloc[0]} - {filtered_results[filtered_results['id']==x]['model_name'].iloc[0]}"
            )
            
            if selected_result_id:
                result_details = filtered_results[filtered_results['id'] == selected_result_id].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Run ID:**", result_details['id'])
                    st.write("**Prompt ID:**", result_details['prompt_id'])
                    st.write("**Category:**", result_details['category'])
                    st.write("**Model:**", result_details['model_name'])
                    st.write("**Status:**", result_details['status'])
                
                with col2:
                    st.write("**Execution Time:**", f"{result_details['execution_time']:.2f}s")
                    st.write("**Timestamp:**", result_details['timestamp'])
                    if 'pass_fail_status' in result_details and pd.notna(result_details['pass_fail_status']):
                        st.write("**Pass/Fail:**", result_details['pass_fail_status'])
                
                st.write("**Model Response:**")
                st.text_area("Response", result_details['response'], height=200, disabled=True)
        
        # Export options
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export to CSV", use_container_width=True):
                csv_data = export_to_csv(filtered_results)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    f"ai_prompt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("📋 Export to JSON", use_container_width=True):
                json_data = export_to_json(filtered_results)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"ai_prompt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
    
    else:
        st.info("No test results found. Run some tests first!")

with tab4:
    st.header("Analysis Dashboard")
    st.markdown("Visual analysis of model performance across categories and time")
    
    # Load data for analysis
    results_df = db_manager.get_execution_results()
    
    if not results_df.empty:
        # Date filtering for analysis
        col1, col2 = st.columns(2)
        with col1:
            analysis_days = st.selectbox(
                "Analysis period",
                options=[7, 14, 30, 60, 90],
                index=2,
                format_func=lambda x: f"Last {x} days"
            )
        
        with col2:
            chart_type = st.selectbox(
                "Chart style",
                options=["Interactive", "Static"],
                index=0
            )
        
        # Filter data for analysis period
        cutoff_date = datetime.now() - timedelta(days=analysis_days)
        analysis_df = results_df[pd.to_datetime(results_df['timestamp']) >= cutoff_date]
        
        if not analysis_df.empty:
            # Success rate by model
            st.subheader("Success Rate by Model")
            
            model_success = analysis_df.groupby('model_name').agg({
                'status': lambda x: (x == 'success').sum() / len(x) * 100
            }).round(2)
            model_success.columns = ['Success Rate (%)']
            
            fig_model = px.bar(
                model_success.reset_index(),
                x='model_name',
                y='Success Rate (%)',
                title="Model Success Rates",
                color='Success Rate (%)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_model, use_container_width=True)
            
            # Success rate by category
            st.subheader("Success Rate by Category")
            
            category_success = analysis_df.groupby('category').agg({
                'status': lambda x: (x == 'success').sum() / len(x) * 100
            }).round(2)
            category_success.columns = ['Success Rate (%)']
            
            fig_category = px.bar(
                category_success.reset_index(),
                x='category',
                y='Success Rate (%)',
                title="Category Success Rates",
                color='Success Rate (%)',
                color_continuous_scale='RdYlGn'
            )
            fig_category.update_xaxis(tickangle=45)
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Performance over time
            st.subheader("Performance Trends Over Time")
            
            # Daily success rates
            analysis_df['date'] = pd.to_datetime(analysis_df['timestamp']).dt.date
            daily_stats = analysis_df.groupby('date').agg({
                'status': lambda x: (x == 'success').sum() / len(x) * 100,
                'execution_time': 'mean'
            }).round(2)
            daily_stats.columns = ['Success Rate (%)', 'Avg Execution Time (s)']
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily_stats.index,
                y=daily_stats['Success Rate (%)'],
                mode='lines+markers',
                name='Success Rate (%)',
                yaxis='y'
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily_stats.index,
                y=daily_stats['Avg Execution Time (s)'],
                mode='lines+markers',
                name='Avg Execution Time (s)',
                yaxis='y2'
            ))
            
            fig_trend.update_layout(
                title="Daily Performance Trends",
                xaxis_title="Date",
                yaxis=dict(title="Success Rate (%)", side="left"),
                yaxis2=dict(title="Execution Time (s)", side="right", overlaying="y"),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Model vs Category heatmap
            st.subheader("Model Performance by Category")
            
            heatmap_data = analysis_df.groupby(['model_name', 'category']).agg({
                'status': lambda x: (x == 'success').sum() / len(x) * 100
            }).round(1)
            heatmap_data.columns = ['Success Rate (%)']
            heatmap_pivot = heatmap_data.reset_index().pivot(
                index='model_name',
                columns='category',
                values='Success Rate (%)'
            )
            
            fig_heatmap = px.imshow(
                heatmap_pivot,
                title="Success Rate Heatmap (Model vs Category)",
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig_heatmap.update_layout(
                xaxis_title="Category",
                yaxis_title="Model"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Execution time distribution
            st.subheader("Execution Time Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    analysis_df,
                    x='execution_time',
                    nbins=30,
                    title="Execution Time Distribution",
                    labels={'execution_time': 'Execution Time (s)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    analysis_df,
                    x='model_name',
                    y='execution_time',
                    title="Execution Time by Model"
                )
                fig_box.update_xaxis(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Test Runs",
                    len(analysis_df),
                    delta=f"Last {analysis_days} days"
                )
            
            with col2:
                overall_success_rate = (analysis_df['status'] == 'success').sum() / len(analysis_df) * 100
                st.metric(
                    "Overall Success Rate",
                    f"{overall_success_rate:.1f}%"
                )
            
            with col3:
                avg_exec_time = analysis_df['execution_time'].mean()
                st.metric(
                    "Average Execution Time",
                    f"{avg_exec_time:.2f}s"
                )
            
            # Detailed statistics table
            st.subheader("Detailed Statistics by Model")
            
            detailed_stats = analysis_df.groupby('model_name').agg({
                'status': ['count', lambda x: (x == 'success').sum()],
                'execution_time': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            detailed_stats.columns = [
                'Total Runs', 'Successful Runs',
                'Avg Time (s)', 'Std Time (s)', 'Min Time (s)', 'Max Time (s)'
            ]
            
            detailed_stats['Success Rate (%)'] = (
                detailed_stats['Successful Runs'] / detailed_stats['Total Runs'] * 100
            ).round(1)
            
            st.dataframe(detailed_stats, use_container_width=True)
        
        else:
            st.info(f"No data available for the last {analysis_days} days. Try expanding the time range or run more tests.")
    
    else:
        st.info("No data available for analysis. Run some tests first!")

# Footer
st.markdown("---")
st.markdown(
    "🛡️ **Ethical AI Prompt Library** - Built for testing LLM safety and robustness | "
    "[Documentation](README.md) | [API Endpoints](/docs)"
)
