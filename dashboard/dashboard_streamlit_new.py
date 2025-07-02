import streamlit as st
import pandas as pd
import altair as alt
import os
from pathlib import Path
import numpy as np
import ast

# Load data from CSV
# Look for the multi_parameter_benchmark_results.csv in checkpoints directory
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
benchmark_file = script_dir.parent / "analyses" / "big_script" / "checkpoints" / "multi_parameter_benchmark_results.csv"

# Fallback to local file if not found in expected location
if not benchmark_file.exists():
    benchmark_file = "multi_parameter_benchmark_results.csv"

# Load the data
data = pd.read_csv(benchmark_file)

# Expand dashboard to full width
st.set_page_config(layout="wide")

# Add a sidebar for navigation
dashboards = ["FP Rate in k Identification", "Execution Time vs. vec_len", "System Type Comparison", "PDF & Example Explorer"]
selected_dashboard = st.sidebar.radio("Select Dashboard:", dashboards)

# Streamlit UI
st.title("Identification Systems Dashboard")
st.subheader(selected_dashboard)

# Handle PDF & Example Explorer separately (different layout)
if selected_dashboard == "PDF & Example Explorer":
    # Load the PDF and examples CSV
    pdf_csv_path = script_dir.parent / "analyses" / "collision" / "pdfs_and_examples.csv"
    
    if not pdf_csv_path.exists():
        st.warning("PDF/example data not found. Please run the collision analysis first.")
        st.info(f"Expected file location: {pdf_csv_path}")
    else:
        # Load the data
        pdf_df = pd.read_csv(pdf_csv_path)
        
        # Parse the string representations of lists into actual lists
        def parse_list(list_str):
            try:
                return ast.literal_eval(list_str)
            except (SyntaxError, ValueError):
                return {}
        
        # Process all columns that contain lists
        pdf_df["msg_pdf"] = pdf_df["msg_pdf"].apply(parse_list)
        pdf_df["examples"] = pdf_df["examples"].apply(parse_list)
        
        # Extract system names from column names
        system_cols = [col for col in pdf_df.columns if col.startswith("tag_pdf_")]
        system_names = [col.replace("tag_pdf_", "") for col in system_cols]
        
        # Parse tag PDFs
        for col in system_cols:
            pdf_df[col] = pdf_df[col].apply(parse_list)
        
        # Controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Controls")
            
            # Select message pattern
            pattern = st.radio("Select Message Pattern:", pdf_df["pattern"].tolist())
            
            # Select system for tag PDF
            system = st.radio("Select System for Tag PDF:", system_names)
            
            # Display metadata
            st.subheader("Pattern Info")
            pattern_data = pdf_df[pdf_df["pattern"] == pattern].iloc[0]
            
            # Display how many examples are available
            num_examples = len(pattern_data["examples"])
            if num_examples > 0:
                st.write(f"{num_examples} example(s) available")
            else:
                st.write("No examples available")
                
            # Display KL divergence and FP rate values
            if f"msg_kl_div" in pattern_data:
                msg_kl = pattern_data["msg_kl_div"]
                st.write(f"Message PDF KL Divergence: {msg_kl:.3f}")
            if f"tag_kl_div_{system}" in pattern_data:
                tag_kl = pattern_data[f"tag_kl_div_{system}"]
                st.write(f"Tag PDF KL Divergence: {tag_kl:.3f}")
            if f"fp_rate_{system}" in pattern_data:
                tag_fp = pattern_data[f"fp_rate_{system}"]
                st.write(f"FP Rate: {tag_fp:.2e}")
        
        # Display area
        with col2:
            # Get the data for the selected pattern
            row = pdf_df[pdf_df["pattern"] == pattern].iloc[0]
            msg_pdf = row["msg_pdf"]
            examples = row["examples"]
            tag_pdf = row[f"tag_pdf_{system}"]
            
            # 1. Display examples on top
            st.subheader(f"Example Messages for '{pattern}' Pattern")
            
            if examples and len(examples) > 0:
                # Convert to numpy array for visualization
                examples_array = np.array(examples)
                
                # Create a DataFrame for Altair visualization
                examples_df = []
                for i, example in enumerate(examples):
                    for j, value in enumerate(example):
                        examples_df.append({
                            'Example': f'Example {i+1}',
                            'Position': j,
                            'Value': value
                        })
                examples_df = pd.DataFrame(examples_df)
                
                # Create heatmap with Altair
                examples_chart = alt.Chart(examples_df).mark_rect().encode(
                    x=alt.X('Position:O', title='Byte Position'),
                    y=alt.Y('Example:O', title=''),
                    color=alt.Color('Value:Q', title='Byte Value', scale=alt.Scale(scheme='viridis')),
                    tooltip=['Example:N', 'Position:O', 'Value:Q']
                ).properties(
                    width=600,
                    height=200
                ).resolve_scale(
                    color='independent'
                )
                
                st.altair_chart(examples_chart, use_container_width=True)
            else:
                st.info("No example messages available for this pattern.")
            
            # 2. Message and Tag PDFs side by side
            col_pdf1, col_pdf2 = st.columns(2)
            
            with col_pdf1:
                st.subheader("Message PDF")
                
                # Create DataFrame for message PDF
                msg_pdf_df = pd.DataFrame({
                    'Symbol': range(256),
                    'Probability': msg_pdf,
                    'Type': 'Message PDF'
                })
                
                # Filter out zeros for better visualization
                msg_pdf_df_filtered = msg_pdf_df[msg_pdf_df['Probability'] > 0]
                
                # Create line chart with Altair
                msg_chart = alt.Chart(msg_pdf_df_filtered).mark_circle(
                    size=20
                ).encode(
                    x=alt.X('Symbol:Q', title='Symbol Value', scale=alt.Scale(domain=[0, 255])),
                    y=alt.Y('Probability:Q', title='Probability'),
                    color=alt.value('#CAE4FF'),
                    tooltip=['Symbol:Q', 'Probability:Q']
                ).properties(
                    width=300,
                    height=250,                    
                )
                
                # Add area fill
                msg_area = alt.Chart(msg_pdf_df).mark_area(
                    opacity=0.15
                ).encode(
                    x=alt.X('Symbol:Q'),
                    y=alt.Y('Probability:Q'),
                    color=alt.value("#CAE4FF")
                )
                
                st.altair_chart((msg_area + msg_chart), use_container_width=True)
            
            with col_pdf2:
                st.subheader(f"Tag PDF ({system})")
                
                # Create DataFrame for tag PDF
                tag_pdf_df = pd.DataFrame({
                    'Symbol': range(256),
                    'Probability': tag_pdf,
                    'Type': f'{system} Tags'
                })
                
                # Filter out zeros for better visualization
                tag_pdf_df_filtered = tag_pdf_df[tag_pdf_df['Probability'] > 0]
                
                # Create line chart with Altair
                tag_chart = alt.Chart(tag_pdf_df_filtered).mark_circle(
                    size=20
                ).encode(
                    x=alt.X('Symbol:Q', title='Symbol Value', scale=alt.Scale(domain=[0, 255])),
                    y=alt.Y('Probability:Q', title='Probability'),
                    color=alt.value("#FF321B"),
                    tooltip=['Symbol:Q', 'Probability:Q']
                ).properties(
                    width=300,
                    height=250
                )
                
                # Add area fill
                tag_area = alt.Chart(tag_pdf_df).mark_area(
                    opacity=0.15
                ).encode(
                    x=alt.X('Symbol:Q'),
                    y=alt.Y('Probability:Q'),
                    color=alt.value('#FF321B')
                )
                
                st.altair_chart((tag_area + tag_chart), use_container_width=True)

# Handle FP Rate in k Identification dashboard
elif selected_dashboard == "FP Rate in k Identification":

    # Filter data for false positive rate
    filtered_data = data[data["test_type"] == "false_positive_rate"]

    # Create three columns: controls and chart
    col1, col2, col3 = st.columns([1, 2, 2])  # Adjust ratio as needed

    with col1:
        st.subheader("Controls")

        # Select system types
        system_types = sorted(data["system_type"].unique())
        # Select system type 1
        selected_system_1 = st.selectbox("Select system type 1:", system_types)
        # Select system type 2
        selected_system_2 = st.selectbox("Select system type 2:", system_types)

        #if selected_system_1 == selected_system_2:
            #st.warning("Please select two different system types for comparison.")
        
        # Filter data by selected system type
        filtered_data_1 = data[data["system_type"] == selected_system_1]
        filtered_data_2 = data[data["system_type"] == selected_system_2]
        
        # Select GF exponent
        # Only show GF exponents present in both datasets
        gf_exps_1 = set(filtered_data_1["gf_exp"].unique())
        gf_exps_2 = set(filtered_data_2["gf_exp"].unique())
        gf_exps = sorted(gf_exps_1 & gf_exps_2)
        if gf_exps:
            selected_gf_exp = st.selectbox("Select GF exponent:", gf_exps)
            filtered_data_1 = filtered_data_1[filtered_data_1["gf_exp"] == selected_gf_exp]
            filtered_data_2 = filtered_data_2[filtered_data_2["gf_exp"] == selected_gf_exp]

        # Select number of tags
        # Only show num_tags present in both datasets
        num_tags_1 = set(filtered_data_1["num_tags"].unique())
        num_tags_2 = set(filtered_data_2["num_tags"].unique())
        num_tags = sorted(num_tags_1 & num_tags_2)
        if num_tags:
            selected_tag = st.radio(
                "Select number of tags:",
                num_tags,
                format_func=lambda x: f"Tags: {x}"
            )
            filtered_data_1 = filtered_data_1[filtered_data_1["num_tags"] == selected_tag]
            filtered_data_2 = filtered_data_2[filtered_data_2["num_tags"] == selected_tag]

        # Select message pattern
        patterns = sorted(filtered_data_1["message_pattern"].unique())
        if patterns:
            selected_pattern = st.selectbox("Message pattern:", patterns)
            filtered_data_1 = filtered_data_1[filtered_data_1["message_pattern"] == selected_pattern]
            filtered_data_2 = filtered_data_2[filtered_data_2["message_pattern"] == selected_pattern]

        if selected_pattern == "random":
            st.markdown("random explanation")
        elif selected_pattern == "low_entropy":
            st.markdown("low_entropy explanation")
        elif selected_pattern == "sparse":
            st.markdown("sparse explanation")
        


        # Pivot data for plotting multiple lines based on test_type
        if not filtered_data_1.empty and not filtered_data_2.empty:

            # combine the two filtered datasets
            filtered_data = pd.concat([filtered_data_1, filtered_data_2])

            pivot_fp_rate = filtered_data.pivot_table(
                index="num_validation_messages",
                columns="system_type",
                values="false_positive_rate"
            ).sort_index()
            
            # Choose code rate column (use code_rate_bulk if available)
            code_rate_col = "code_rate_bulk" if "code_rate_bulk" in filtered_data.columns else "code_rate"
            pivot_code_rate = filtered_data.pivot_table(
                index="num_validation_messages",
                columns="system_type",
                values=code_rate_col
            ).sort_index()


    with col2:
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

            # Display metrics
            st.markdown(
                f"<span style='font-size:14px;'>vec_len range: {filtered_data['vec_len'].min()} - {filtered_data['vec_len'].max()}</span>",
                unsafe_allow_html=True
            )
            avg_messages = filtered_data['num_messages'].mean()
            if avg_messages > 0:
                power_of_ten = int(np.floor(np.log10(avg_messages)))
            else:
                power_of_ten = 0
            st.markdown(
                f"<span style='font-size:14px;'>avg messages: 10<sup>{power_of_ten}</sup></span>",
                unsafe_allow_html=True
            )
            
        else:
            st.write("No data available with current filter settings.")

    with col3:
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

    

# Handle execution time vs. vector length dashboard
elif selected_dashboard == "Execution Time vs. vec_len":
    # Filter data for execution time
    filtered_data = data[data["test_type"] == "execution_time"]

    # Create three columns: controls and chart
    col1, col2, col3 = st.columns([1, 2, 2])  # Adjust ratio as needed

    with col1:
        st.subheader("Controls")

        # Select system type
        system_types = sorted(filtered_data["system_type"].unique())
        selected_system = st.selectbox("Select system type:", system_types)
        
        # Filter data by selected system type
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

    with col2:
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

    with col3:
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

# Handle system type comparison dashboard
elif selected_dashboard == "System Type Comparison":
    # Filter data for system type comparison
    filtered_data = data[data["test_type"] == "execution_time"]

    # Create three columns: controls and chart
    col1, col2, col3 = st.columns([1, 2, 2])  # Adjust ratio as needed

    with col1:
        st.subheader("Controls")

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

    with col2:
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

    with col3:
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