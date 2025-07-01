#!/usr/bin/env python3
"""
Streamlit dashboard for comprehensive identification systems analysis.

This dashboard visualizes results from the multi-parameter benchmark script,
showing execution time, false positive rates, and system comparisons across
multiple dimensions.
"""

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path

# 🔧 Expand dashboard to full width
st.set_page_config(page_title="ID Systems Benchmark Dashboard", layout="wide")

# Load data with error handling
@st.cache_data
def load_data():
    """Load the benchmark results CSV with caching."""
    try:
        # Try to find the CSV file in common locations
        possible_paths = [
            "multi_parameter_benchmark_results.csv",
            "analyses/big_script/checkpoints/multi_parameter_benchmark_results.csv",
            "../analyses/big_script/checkpoints/multi_parameter_benchmark_results.csv",
            "checkpoints/multi_parameter_benchmark_results.csv"
        ]
        
        data = None
        for path in possible_paths:
            try:
                data = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            st.error("❌ Could not find the benchmark results CSV file. Please ensure it exists in one of the expected locations.")
            st.stop()
        
        # Convert tag_pos from string representation to actual list length
        if 'tag_pos' in data.columns:
            data['tag_pos_str'] = data['tag_pos'].astype(str)
        
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Load the data
data = load_data()

# Add a sidebar for navigation
st.sidebar.title("🎛️ Navigation")
dashboards = [
    "📊 Overview", 
    "⚡ Execution Time Analysis", 
    "🎯 False Positive Rate Analysis",
    "🔄 System Comparison",
    "📈 Multi-dimensional Analysis"
]
selected_dashboard = st.sidebar.selectbox("Select Dashboard:", dashboards)

# Display basic data info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("📋 **Dataset Info**")
st.sidebar.markdown(f"**Total Records:** {len(data):,}")
st.sidebar.markdown(f"**Systems:** {', '.join(sorted(data['system_type'].unique()))}")
st.sidebar.markdown(f"**GF Exponents:** {', '.join(map(str, sorted(data['gf_exp'].unique())))}")
st.sidebar.markdown(f"**Test Types:** {', '.join(sorted(data['test_type'].unique()))}")

# Main title
st.title("🔬 Identification Systems Benchmark Dashboard")
st.markdown(f"**Current View:** {selected_dashboard}")

# --- OVERVIEW DASHBOARD ---
if selected_dashboard == "📊 Overview":
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", f"{len(data):,}")
        st.metric("Systems Tested", len(data['system_type'].unique()))
    
    with col2:
        exec_tests = len(data[data['test_type'] == 'execution_time'])
        fp_tests = len(data[data['test_type'] == 'false_positive_rate'])
        st.metric("Execution Time Tests", f"{exec_tests:,}")
        st.metric("FP Rate Tests", f"{fp_tests:,}")
    
    with col3:
        st.metric("Vector Lengths Tested", len(data['vec_len'].unique()))
        st.metric("GF Exponents", len(data['gf_exp'].unique()))
    
    with col4:
        avg_exec_time = data['avg_execution_time_ms'].mean()
        avg_fp_rate = data['false_positive_rate'].mean()
        st.metric("Avg Execution Time", f"{avg_exec_time:.2f} ms")
        st.metric("Avg FP Rate", f"{avg_fp_rate:.6f}")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Test Distribution by System Type")
        system_counts = data['system_type'].value_counts().reset_index()
        system_counts.columns = ['System', 'Count']
        
        chart = alt.Chart(system_counts).mark_bar().encode(
            x=alt.X('System:N', sort='-y'),
            y=alt.Y('Count:Q'),
            color=alt.Color('System:N', legend=None),
            tooltip=['System:N', 'Count:Q']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Test Distribution by GF Exponent")
        gf_counts = data['gf_exp'].value_counts().reset_index()
        gf_counts.columns = ['GF_EXP', 'Count']
        
        chart = alt.Chart(gf_counts).mark_bar().encode(
            x=alt.X('GF_EXP:O'),
            y=alt.Y('Count:Q'),
            color=alt.Color('GF_EXP:O', legend=None),
            tooltip=['GF_EXP:O', 'Count:Q']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# --- EXECUTION TIME ANALYSIS ---
elif selected_dashboard == "⚡ Execution Time Analysis":
    st.markdown("### ⚡ Execution Time Performance Analysis")
    
    # Filter to execution time tests only
    exec_data = data[data['test_type'] == 'execution_time'].copy()
    
    if exec_data.empty:
        st.warning("No execution time data available.")
        st.stop()
    
    # Controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("🎛️ **Controls**")
    
    # System selection
    systems = sorted(exec_data['system_type'].unique())
    selected_systems = st.sidebar.multiselect(
        "Select Systems:", 
        systems, 
        default=systems[:3] if len(systems) > 3 else systems
    )
    
    # GF exponent selection
    gf_exps = sorted(exec_data['gf_exp'].unique())
    selected_gf_exp = st.sidebar.selectbox("Select GF Exponent:", gf_exps)
    
    # Tag count selection
    tag_counts = sorted(exec_data['num_tags'].unique())
    selected_tag_count = st.sidebar.selectbox("Select Tag Count:", tag_counts)
    
    # Filter data
    filtered_data = exec_data[
        (exec_data['system_type'].isin(selected_systems)) &
        (exec_data['gf_exp'] == selected_gf_exp) &
        (exec_data['num_tags'] == selected_tag_count)
    ]
    
    if filtered_data.empty:
        st.warning("No data matches the selected criteria.")
        st.stop()
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Execution Time vs Vector Length")
        
        # Filter out zero or negative values for log scale
        exec_filtered = filtered_data[
            (filtered_data['avg_execution_time_ms'] > 0) & 
            (filtered_data['vec_len'] > 0)
        ]
        
        if exec_filtered.empty:
            st.info("No positive execution times to display on log scale.")
        else:
            chart = alt.Chart(exec_filtered).mark_line(point=True).encode(
                x=alt.X('vec_len:Q', 
                       scale=alt.Scale(type='log', base=2),
                       title='Vector Length (log scale)'),
                y=alt.Y('avg_execution_time_ms:Q', 
                       scale=alt.Scale(type='log'),
                       title='Avg Execution Time (ms, log scale)'),
                color=alt.Color('system_type:N', title='System'),
                tooltip=['system_type:N', 'vec_len:Q', 'avg_execution_time_ms:Q', 'gf_exp:Q', 'num_tags:Q']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Throughput Comparison")
        
        # Filter out zero or negative values and aggregate properly
        throughput_filtered = filtered_data[filtered_data['throughput_msgs_per_sec'] > 0]
        
        if throughput_filtered.empty:
            st.info("No positive throughput values to display.")
        else:
            # Group by system and calculate mean to avoid overlapping points
            avg_throughput = throughput_filtered.groupby('system_type').agg({
                'throughput_msgs_per_sec': 'mean',
                'vec_len': 'mean',  # Include other relevant info
                'gf_exp': lambda x: ', '.join(map(str, sorted(x.unique()))),
                'num_tags': lambda x: ', '.join(map(str, sorted(x.unique())))
            }).reset_index()
            
            chart = alt.Chart(avg_throughput).mark_bar().encode(
                x=alt.X('system_type:N', title='System Type'),
                y=alt.Y('throughput_msgs_per_sec:Q', title='Avg Throughput (msgs/sec)'),
                color=alt.Color('system_type:N', legend=None),
                tooltip=['system_type:N', 'throughput_msgs_per_sec:Q', 'gf_exp:N', 'num_tags:N']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
    
    # Detailed table
    st.markdown("#### 📋 Detailed Performance Metrics")
    display_cols = ['system_type', 'vec_len', 'avg_execution_time_ms', 'min_execution_time_ms', 
                   'max_execution_time_ms', 'std_execution_time_ms', 'throughput_msgs_per_sec']
    st.dataframe(filtered_data[display_cols].round(3), use_container_width=True)

# --- FALSE POSITIVE RATE ANALYSIS ---
elif selected_dashboard == "🎯 False Positive Rate Analysis":
    st.markdown("### 🎯 False Positive Rate Analysis")
    
    # Filter to FP rate tests only
    fp_data = data[data['test_type'] == 'false_positive_rate'].copy()
    
    if fp_data.empty:
        st.warning("No false positive rate data available.")
        st.stop()
    
    # Controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("🎛️ **Controls**")
    
    # System selection
    systems = sorted(fp_data['system_type'].unique())
    selected_systems = st.sidebar.multiselect(
        "Select Systems:", 
        systems, 
        default=systems
    )
    
    # GF exponent selection
    gf_exps = sorted(fp_data['gf_exp'].unique())
    selected_gf_exp = st.sidebar.selectbox("Select GF Exponent:", gf_exps)
    
    # Tag count selection
    tag_counts = sorted(fp_data['num_tags'].unique())
    selected_tag_count = st.sidebar.selectbox("Select Tag Count:", tag_counts)
    
    # Message pattern selection
    patterns = sorted(fp_data['message_pattern'].unique())
    selected_pattern = st.sidebar.selectbox("Select Message Pattern:", patterns)
    
    # Filter data
    filtered_data = fp_data[
        (fp_data['system_type'].isin(selected_systems)) &
        (fp_data['gf_exp'] == selected_gf_exp) &
        (fp_data['num_tags'] == selected_tag_count) &
        (fp_data['message_pattern'] == selected_pattern)
    ]
    
    if filtered_data.empty:
        st.warning("No data matches the selected criteria.")
        st.stop()
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 FP Rate vs Validation Messages")
        
        # Filter out zero values for log scale
        fp_filtered = filtered_data[filtered_data['false_positive_rate'] > 0]
        
        if fp_filtered.empty:
            st.info("No positive false positive rates to display on log scale.")
        else:
            chart = alt.Chart(fp_filtered).mark_line(point=True).encode(
                x=alt.X('num_validation_messages:Q', title='Number of Validation Messages'),
                y=alt.Y('false_positive_rate:Q', 
                       scale=alt.Scale(type='log'),
                       title='False Positive Rate (log scale)'),
                color=alt.Color('system_type:N', title='System'),
                tooltip=['system_type:N', 'num_validation_messages:Q', 'false_positive_rate:Q', 'num_tags:Q', 'gf_exp:Q']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 FP Rate vs Tag Count")
        
        if len(filtered_data['num_tags'].unique()) > 1:
            # Filter out zero values for log scale
            fp_filtered = filtered_data[filtered_data['false_positive_rate'] > 0]
            
            if fp_filtered.empty:
                st.info("No positive false positive rates to display on log scale.")
            else:
                chart = alt.Chart(fp_filtered).mark_line(point=True).encode(
                    x=alt.X('num_tags:Q', title='Number of Tags'),
                    y=alt.Y('false_positive_rate:Q', 
                           scale=alt.Scale(type='log'),
                           title='False Positive Rate (log scale)'),
                    color=alt.Color('system_type:N', title='System'),
                    tooltip=['system_type:N', 'num_tags:Q', 'false_positive_rate:Q', 'num_validation_messages:Q', 'gf_exp:Q']
                ).properties(height=400)
                
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Only one tag count available in filtered data.")
    
    # Code rate analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Code Rate (Bulk) vs Validation Messages")
        st.caption("Code rate depends on number of tags - points may overlap for same system with different tag counts")
        
        # Filter out zero or negative values
        code_filtered = filtered_data[filtered_data['code_rate_bulk'] > 0]
        
        if code_filtered.empty:
            st.info("No positive code rates to display.")
        else:
            chart = alt.Chart(code_filtered).mark_circle(size=60).encode(
                x=alt.X('num_validation_messages:Q', title='Number of Validation Messages'),
                y=alt.Y('code_rate_bulk:Q', title='Code Rate (Bulk)'),
                color=alt.Color('system_type:N', title='System'),
                shape=alt.Shape('num_tags:O', title='Tag Count'),
                tooltip=['system_type:N', 'num_validation_messages:Q', 'code_rate_bulk:Q', 'num_tags:Q', 'gf_exp:Q', 'vec_len:Q']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Code Rate (Subsequently) vs Validation Messages")
        st.caption("Code rate depends on number of tags - different shapes show tag counts")
        
        # Filter out zero or negative values
        code_filtered = filtered_data[filtered_data['code_rate_subsequently'] > 0]
        
        if code_filtered.empty:
            st.info("No positive code rates to display.")
        else:
            chart = alt.Chart(code_filtered).mark_circle(size=60).encode(
                x=alt.X('num_validation_messages:Q', title='Number of Validation Messages'),
                y=alt.Y('code_rate_subsequently:Q', title='Code Rate (Subsequently)'),
                color=alt.Color('system_type:N', title='System'),
                shape=alt.Shape('num_tags:O', title='Tag Count'),
                tooltip=['system_type:N', 'num_validation_messages:Q', 'code_rate_subsequently:Q', 'num_tags:Q', 'gf_exp:Q', 'vec_len:Q']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)

# --- SYSTEM COMPARISON ---
elif selected_dashboard == "🔄 System Comparison":
    st.markdown("### 🔄 Cross-System Performance Comparison")
    
    # Controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("🎛️ **Comparison Controls**")
    
    # Metric selection
    comparison_metrics = {
        'avg_execution_time_ms': 'Average Execution Time (ms)',
        'false_positive_rate': 'False Positive Rate',
        'throughput_msgs_per_sec': 'Throughput (msgs/sec)',
        'code_rate_bulk': 'Code Rate (Bulk)',
        'code_rate_subsequently': 'Code Rate (Subsequently)'
    }
    
    selected_metric = st.sidebar.selectbox(
        "Select Metric for Comparison:", 
        list(comparison_metrics.keys()),
        format_func=lambda x: comparison_metrics[x]
    )
    
    # Filter controls
    test_types = sorted(data['test_type'].unique())
    selected_test_type = st.sidebar.selectbox("Select Test Type:", test_types)
    
    gf_exps = sorted(data['gf_exp'].unique())
    selected_gf_exp = st.sidebar.selectbox("Select GF Exponent:", gf_exps)
    
    # Filter data
    comparison_data = data[
        (data['test_type'] == selected_test_type) &
        (data['gf_exp'] == selected_gf_exp)
    ].copy()
    
    if comparison_data.empty:
        st.warning("No data matches the selected criteria.")
        st.stop()
    
    # System performance heatmap
    st.markdown(f"#### 🔥 System Performance Heatmap: {comparison_metrics[selected_metric]}")
    
    # Create pivot table for heatmap
    if selected_test_type == 'execution_time':
        pivot_data = comparison_data.pivot_table(
            values=selected_metric,
            index='system_type',
            columns='vec_len',
            aggfunc='mean'
        )
    else:
        pivot_data = comparison_data.pivot_table(
            values=selected_metric,
            index='system_type',
            columns='num_validation_messages',
            aggfunc='mean'
        )
    
    # Display heatmap using Streamlit's built-in capabilities
    st.dataframe(pivot_data.round(6), use_container_width=True)
    
    # Bar comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 📊 Average {comparison_metrics[selected_metric]} by System")
        
        avg_by_system = comparison_data.groupby('system_type')[selected_metric].mean().reset_index()
        avg_by_system = avg_by_system.sort_values(selected_metric)
        
        chart = alt.Chart(avg_by_system).mark_bar().encode(
            x=alt.X(f'{selected_metric}:Q', title=comparison_metrics[selected_metric]),
            y=alt.Y('system_type:N', sort='-x', title='System Type'),
            color=alt.Color('system_type:N', legend=None),
            tooltip=['system_type:N', f'{selected_metric}:Q']
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Performance Distribution")
        
        chart = alt.Chart(comparison_data).mark_boxplot().encode(
            x=alt.X('system_type:N', title='System Type'),
            y=alt.Y(f'{selected_metric}:Q', title=comparison_metrics[selected_metric]),
            color=alt.Color('system_type:N', legend=None)
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)

# --- MULTI-DIMENSIONAL ANALYSIS ---
elif selected_dashboard == "📈 Multi-dimensional Analysis":
    st.markdown("### 📈 Multi-dimensional Parameter Analysis")
    
    # Controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("🎛️ **Advanced Controls**")
    
    # Test type selection
    test_types = sorted(data['test_type'].unique())
    selected_test_type = st.sidebar.selectbox("Select Test Type:", test_types)
    
    # Filter data by test type
    filtered_data = data[data['test_type'] == selected_test_type].copy()
    
    # Remove rows with zero or negative values that would cause issues with log scales
    numeric_columns_to_filter = ['avg_execution_time_ms', 'false_positive_rate', 'throughput_msgs_per_sec', 'vec_len']
    for col in numeric_columns_to_filter:
        if col in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[col] > 0]
    
    # Axis selections for scatter plot
    numeric_columns = ['gf_exp', 'vec_len', 'num_tags', 'num_validation_messages', 
                      'avg_execution_time_ms', 'false_positive_rate', 'throughput_msgs_per_sec',
                      'code_rate_bulk', 'code_rate_subsequently']
    
    # Remove columns that don't exist in the current test type
    available_columns = [col for col in numeric_columns if col in filtered_data.columns and filtered_data[col].notna().any()]
    
    x_axis = st.sidebar.selectbox("X-Axis:", available_columns, index=0)
    y_axis = st.sidebar.selectbox("Y-Axis:", available_columns, index=min(1, len(available_columns)-1))
    
    # Color and size encodings
    categorical_columns = ['system_type', 'message_pattern'] if selected_test_type == 'false_positive_rate' else ['system_type']
    color_by = st.sidebar.selectbox("Color by:", categorical_columns)
    
    size_by = st.sidebar.selectbox("Size by:", ['None'] + available_columns)
    
    # Create the multi-dimensional scatter plot
    st.markdown(f"#### 🎯 {y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}")
    st.caption("ℹ️ Points may overlap when systems have identical parameter combinations. Hover for details including tag count and validation messages.")
    
    # Base chart
    chart = alt.Chart(filtered_data).mark_circle(size=100).encode(
        x=alt.X(f'{x_axis}:Q', 
               scale=alt.Scale(type='log', zero=False) if ('time' in x_axis or x_axis == 'vec_len' or 'rate' in x_axis) and filtered_data[x_axis].min() > 0 else alt.Scale(),
               title=x_axis.replace('_', ' ').title()),
        y=alt.Y(f'{y_axis}:Q',
               scale=alt.Scale(type='log', zero=False) if ('time' in y_axis or 'rate' in y_axis) and filtered_data[y_axis].min() > 0 else alt.Scale(),
               title=y_axis.replace('_', ' ').title()),
        color=alt.Color(f'{color_by}:N', title=color_by.replace('_', ' ').title()),
        tooltip=['system_type:N', f'{x_axis}:Q', f'{y_axis}:Q', 'gf_exp:Q', 'num_tags:Q', 'num_validation_messages:Q']
    )
    
    # Add size encoding if selected
    if size_by != 'None':
        chart = chart.encode(
            size=alt.Size(f'{size_by}:Q', 
                         scale=alt.Scale(range=[50, 500]),
                         title=size_by.replace('_', ' ').title())
        )
    
    chart = chart.properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    
    # Correlation matrix for numeric variables
    st.markdown("#### 📊 Correlation Analysis")
    
    # Select only numeric columns for correlation
    numeric_data = filtered_data[available_columns].select_dtypes(include=[np.number])
    
    if len(numeric_data.columns) > 1:
        correlation_matrix = numeric_data.corr()
        
        # Create a correlation heatmap using Altair
        corr_data = correlation_matrix.reset_index().melt('index')
        corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']
        
        heatmap = alt.Chart(corr_data).mark_rect().encode(
            x=alt.X('Variable 1:O', title=''),
            y=alt.Y('Variable 2:O', title=''),
            color=alt.Color('Correlation:Q', 
                           scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                           title='Correlation'),
            tooltip=['Variable 1:O', 'Variable 2:O', 'Correlation:Q']
        ).properties(
            width=400,
            height=400
        )
        
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Not enough numeric variables for correlation analysis.")

# Footer
st.markdown("---")
st.markdown("🔬 **Identification Systems Benchmark Dashboard** | Built with Streamlit & Altair")