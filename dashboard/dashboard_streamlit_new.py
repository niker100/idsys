import streamlit as st
import pandas as pd
import altair as alt
import os
from pathlib import Path

# 📂 Load data from CSV
# Look for the multi_parameter_benchmark_results.csv in checkpoints directory
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
benchmark_file = script_dir.parent / "analyses" / "big_script" / "checkpoints" / "multi_parameter_benchmark_results.csv"

# Fallback to local file if not found in expected location
if not benchmark_file.exists():
    benchmark_file = "multi_parameter_benchmark_results.csv"

# Load the data
data = pd.read_csv(benchmark_file)

# 🔧 Expand dashboard to full width
st.set_page_config(layout="wide")

# Add a sidebar for navigation
dashboards = ["Code Rate vs. FP Rate Tradeoff", "Execution Time vs. vec_len", "System Type Comparison"]
selected_dashboard = st.sidebar.radio("Select Dashboard:", dashboards)

# Streamlit UI
st.title("Identification Systems Dashboard")
st.subheader(selected_dashboard)

# Create three columns: controls and chart
col1, col2, col3 = st.columns([1, 2, 2])  # Adjust ratio as needed

# Filter by test_type
if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
    # We need false_positive_rate data for this dashboard
    filtered_data = data[data["test_type"] == "false_positive_rate"]
elif selected_dashboard == "Execution Time vs. vec_len":
    # We need execution_time data for this dashboard
    filtered_data = data[data["test_type"] == "execution_time"]
else:
    # For system comparison, use execution time data as default
    filtered_data = data[data["test_type"] == "execution_time"]

# --- Controls Section (Left Column) ---
with col1:
    st.subheader("Controls")

    # Controls for "Code Rate vs. FP Rate Tradeoff" dashboard
    if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
        # Select system type
        system_types = sorted(filtered_data["system_type"].unique())
        selected_system = st.selectbox("Select system type:", system_types)
        
        # Further filter by system type
        filtered_data = filtered_data[filtered_data["system_type"] == selected_system]
        
        # Select GF exponent
        gf_exps = sorted(filtered_data["gf_exp"].unique())
        if gf_exps:
            selected_gf_exp = st.selectbox("Select GF exponent:", gf_exps)
            filtered_data = filtered_data[filtered_data["gf_exp"] == selected_gf_exp]
        
        # Select number of tags
        num_tags = sorted(filtered_data["num_tags"].unique())
        if num_tags:
            selected_tag = st.radio(
                "Select number of tags:",
                num_tags,
                format_func=lambda x: f"Tags: {x}"
            )
            filtered_data = filtered_data[filtered_data["num_tags"] == selected_tag]
        
        # Select message pattern
        patterns = sorted(filtered_data["message_pattern"].unique())
        if patterns:
            selected_pattern = st.selectbox("Message pattern:", patterns)
            filtered_data = filtered_data[filtered_data["message_pattern"] == selected_pattern]
        
        # Pivot data for plotting multiple lines based on test_type
        if not filtered_data.empty:
            pivot_fp_rate = filtered_data.pivot_table(
                index="num_validation_messages",
                columns="vec_len",
                values="false_positive_rate"
            ).sort_index()
            
            # Choose code rate column (use code_rate_bulk if available)
            code_rate_col = "code_rate_bulk" if "code_rate_bulk" in filtered_data.columns else "code_rate"
            pivot_code_rate = filtered_data.pivot_table(
                index="num_validation_messages",
                columns="vec_len",
                values=code_rate_col
            ).sort_index()
            
            # Display metrics
            st.markdown(
                f"<span style='font-size:14px;'>vec_len range: {filtered_data['vec_len'].min()} - {filtered_data['vec_len'].max()}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span style='font-size:14px;'>avg messages: {filtered_data['num_messages'].mean():.0f}</span>",
                unsafe_allow_html=True
            )

    # Controls for "Execution Time vs. vec_len" dashboard
    if selected_dashboard == "Execution Time vs. vec_len":
        # Select system type
        system_types = sorted(filtered_data["system_type"].unique())
        selected_system = st.selectbox("Select system type:", system_types)
        
        # Filter by system type
        filtered_data = filtered_data[filtered_data["system_type"] == selected_system]
        
        # Select GF exponent
        gf_exps = sorted(filtered_data["gf_exp"].unique())
        if gf_exps:
            selected_gf_exp = st.radio(
                "Select GF exponent:",
                gf_exps,
                format_func=lambda x: f"GF(2^{x})"
            )
            filtered_data = filtered_data[filtered_data["gf_exp"] == selected_gf_exp]
        
        # Select number of tags
        num_tags = sorted(filtered_data["num_tags"].unique())
        if num_tags:
            selected_num_tags = st.selectbox("Number of tags:", num_tags)
            filtered_data = filtered_data[filtered_data["num_tags"] == selected_num_tags]
        
        # Display metrics
        st.markdown(
            f"<span style='font-size:14px;'>number of messages: {filtered_data['num_messages'].mean():.0f}</span>",
            unsafe_allow_html=True
        )
    
    # Controls for "System Type Comparison" dashboard
    if selected_dashboard == "System Type Comparison":
        # Select GF exponent
        gf_exps = sorted(filtered_data["gf_exp"].unique())
        if gf_exps:
            selected_gf_exp = st.radio(
                "Select GF exponent:",
                gf_exps,
                format_func=lambda x: f"GF(2^{x})"
            )
            filtered_data = filtered_data[filtered_data["gf_exp"] == selected_gf_exp]
        
        # Select number of tags
        num_tags = sorted(filtered_data["num_tags"].unique())
        if num_tags:
            selected_num_tags = st.selectbox("Number of tags:", num_tags)
            filtered_data = filtered_data[filtered_data["num_tags"] == selected_num_tags]
            
        # Filter to single-tag systems for fair comparison
        filtered_data = filtered_data[filtered_data["num_tags"] == selected_num_tags]

# --- Chart and Metrics Section (Middle Column) ---
with col2:
    if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
        st.subheader("False Positive Rate")
        if not filtered_data.empty and 'pivot_fp_rate' in locals():
            # Line chart for false positive rate by vector length
            chart_data = pivot_fp_rate.reset_index().melt(
                'num_validation_messages',
                var_name='vec_len',
                value_name='false_positive_rate'
            )
            
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("num_validation_messages:Q", title="Number of Validation Messages"),
                    y=alt.Y("false_positive_rate:Q", title="False Positive Rate"),
                    color=alt.Color("vec_len:N", legend=alt.Legend(title="Vector Length", orient="bottom")),
                    tooltip=["num_validation_messages:Q", "vec_len:N", "false_positive_rate:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available with current filter settings.")

    if selected_dashboard == "Execution Time vs. vec_len":
        st.subheader("Execution Time vs. Vector Length")
        
        if not filtered_data.empty:
            # Line chart for execution time vs. vec_len
            chart = (
                alt.Chart(filtered_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("vec_len:Q", title="Vector Length", scale=alt.Scale(type='log', base=2)),
                    y=alt.Y("avg_execution_time_ms:Q", title="Average Execution Time (ms)", scale=alt.Scale(type='log')),
                    tooltip=["vec_len:Q", "avg_execution_time_ms:Q", "system_type:N", "num_tags:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available for the selected system and parameters.")
    
    if selected_dashboard == "System Type Comparison":
        st.subheader("Execution Time by System Type")
        
        if not filtered_data.empty:
            # Prepare data for comparison chart - filter for specific vec_len
            vec_lengths = sorted(filtered_data["vec_len"].unique())
            if vec_lengths:
                selected_vec_len = st.select_slider(
                    "Select vector length:",
                    options=vec_lengths
                )
                comparison_data = filtered_data[filtered_data["vec_len"] == selected_vec_len]
                
                # Bar chart comparing system types
                chart = (
                    alt.Chart(comparison_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("system_type:N", title="System Type"),
                        y=alt.Y("avg_execution_time_ms:Q", title="Avg Execution Time (ms)"),
                        color="system_type:N",
                        tooltip=["system_type:N", "avg_execution_time_ms:Q", "vec_len:Q"]
                    )
                    .properties(width=700, height=400)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No vector length data available.")
        else:
            st.write("No data available for comparison.")

# --- Chart and Metrics Section (Right Column) ---
with col3:
    if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
        st.subheader("Code Rate")
        if not filtered_data.empty and 'pivot_code_rate' in locals():
            # Line chart for code rate by vector length
            chart_data = pivot_code_rate.reset_index().melt(
                'num_validation_messages',
                var_name='vec_len',
                value_name='code_rate'
            )
            
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("num_validation_messages:Q", title="Number of Validation Messages"),
                    y=alt.Y("code_rate:Q", title="Code Rate"),
                    color=alt.Color("vec_len:N", legend=alt.Legend(title="Vector Length", orient="bottom")),
                    tooltip=["num_validation_messages:Q", "vec_len:N", "code_rate:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available with current filter settings.")

    if selected_dashboard == "Execution Time vs. vec_len":
        st.subheader("Throughput vs. Vector Length")
        
        if not filtered_data.empty:
            # Line chart for throughput vs. vec_len
            chart = (
                alt.Chart(filtered_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("vec_len:Q", title="Vector Length", scale=alt.Scale(type='log', base=2)),
                    y=alt.Y("throughput_msgs_per_sec:Q", title="Messages Per Second", scale=alt.Scale(type='log')),
                    tooltip=["vec_len:Q", "throughput_msgs_per_sec:Q", "system_type:N"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No throughput data available.")
    
    if selected_dashboard == "System Type Comparison":
        st.subheader("Throughput by System Type")
        
        if not filtered_data.empty:
            # Use the same vec_len as middle column
            if 'selected_vec_len' in locals() and 'comparison_data' in locals():
                # Bar chart comparing throughput
                chart = (
                    alt.Chart(comparison_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("system_type:N", title="System Type"),
                        y=alt.Y("throughput_msgs_per_sec:Q", title="Messages Per Second (throughput)"),
                        color="system_type:N",
                        tooltip=["system_type:N", "throughput_msgs_per_sec:Q", "vec_len:Q"]
                    )
                    .properties(width=700, height=400)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No vector length selected.")
        else:
            st.write("No throughput data available.")

# Add a footer with data info
st.sidebar.markdown("---")
st.sidebar.info(f"Data source: {os.path.basename(benchmark_file)}")