import streamlit as st
import pandas as pd
import altair as alt

# 📂 Load data from CSV
data_multi_tag = pd.read_csv("multi_tag_multi_message_results.csv")
data_execution_time = pd.read_csv("execution_time_results.csv")

# 🔧 Expand dashboard to full width
st.set_page_config(layout="wide")

# Add a sidebar for navigation
dashboards = ["Code Rate vs. FP Rate Tradeoff", "Execution Time vs. vec_len"]
selected_dashboard = st.sidebar.radio("Select Dashboard:", dashboards)

# Streamlit UI
st.title("Identification Systems Dashboard")
st.subheader(selected_dashboard)

# Create three columns: controls and chart
col1, col2, col3 = st.columns([1, 2, 2])  # Adjust ratio as needed
# --- Controls and Charts Section ---

# Initialize filtered_data variable
filtered_data = []

# --- Controls Section (Left Column) ---
with col1:
    st.subheader("Controls")

    # Controls for "Code Rate vs. FP Rate Tradeoff" dashboard
    if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
        # Select number of tags from unique values in the dataset
        num_tags = sorted(data_multi_tag["num_tags"].unique())
        selected_tag = st.radio(
            "Select number of tags:",
            num_tags,
            format_func=lambda x: f"Tag: {x}"
        )

        # Filter data based on selected tag
        filtered_data = data_multi_tag[
            data_multi_tag["num_tags"] == selected_tag
        ]

        # Pivot data for plotting multiple lines (test types) for FP rate and code rate
        pivot_fp_rate = (
            filtered_data.pivot_table(
                index="num_validation_messages",
                columns="test_type",
                values="false_positive_rate"
            ).sort_index()
        )
        pivot_code_rate = (
            filtered_data.pivot_table(
                index="num_validation_messages",
                columns="test_type",
                values="code_rate"
            )
        )

        # Display metrics for vec_len and number of messages
        st.markdown(
            f"<span style='font-size:14px;'>vec_len: {filtered_data['vec_len'].mean():.0f}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<span style='font-size:14px;'>number of messages: {filtered_data['num_messages'].mean():.0f}</span>",
            unsafe_allow_html=True
        )

    # Controls for "Execution Time vs. vec_len" dashboard
    if selected_dashboard == "Execution Time vs. vec_len":
        # Select gf_exp from unique values in the dataset
        gf_exps = sorted(data_execution_time["gf_exp"].unique())
        selected_gf_exp = st.radio(
            "Select gf_exp:",
            gf_exps,
            format_func=lambda x: f"Tag: {x}"
        )

        # Filter data based on selected gf_exp
        filtered_data = data_execution_time[
            data_execution_time["gf_exp"] == selected_gf_exp
        ]

        # Display metric for number of messages
        st.markdown(
            f"<span style='font-size:14px;'>number of messages: {filtered_data['num_messages'].mean():.0f}</span>",
            unsafe_allow_html=True
        )

# --- Chart and Metrics Section (Middle Column) ---
with col2:
    if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
        st.subheader("false positive rate")
        # Line chart for false positive rate by test type
        chart = (
            alt.Chart(
                pivot_fp_rate.reset_index().melt(
                    'num_validation_messages',
                    var_name='test_type',
                    value_name='false_positive_rate'
                )
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("num_validation_messages:Q", title="Number of Validation Messages"),
                y=alt.Y("false_positive_rate:Q", title="False Positive Rate"),
                color=alt.Color("test_type:N", legend=alt.Legend(title="Test Type", orient="bottom")),
                tooltip=["num_validation_messages:Q", "test_type:N", "false_positive_rate:Q"]
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    if selected_dashboard == "Execution Time vs. vec_len":
        st.subheader("Execution Time vs. vec_len")
        # Select encoder for chart from unique values in filtered data
        encoders = sorted(filtered_data["encoder"].unique())
        selected_encoder = st.selectbox("Select encoder:", encoders, key="encoder")

        # Filter data for the selected encoder
        encoder_data = filtered_data[filtered_data["encoder"] == selected_encoder]

        if not encoder_data.empty:
            # Line chart for execution time vs. vec_len
            chart = (
                alt.Chart(encoder_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("vec_len:Q", title="vec_len"),
                    y=alt.Y("avg_execution_time_ms:Q", title="Average Execution Time (ms)"),
                    tooltip=["vec_len:T", "encoder:N", "avg_execution_time_ms:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available for the selected encoder.")

# --- Chart and Metrics Section (Right Column) ---
with col3:
    if selected_dashboard == "Code Rate vs. FP Rate Tradeoff":
        st.subheader("code rate")
        # Line chart for code rate by test type
        chart = (
            alt.Chart(
                pivot_code_rate.reset_index().melt(
                    'num_validation_messages',
                    var_name='test_type',
                    value_name='code_rate'
                )
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("num_validation_messages:Q", title="Number of Validation Messages"),
                y=alt.Y("code_rate:Q", title="Code Rate"),
                color=alt.Color("test_type:N", legend=alt.Legend(title="Test Type", orient="bottom")),
                tooltip=["num_validation_messages:Q", "test_type:N", "code_rate:Q"]
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    if selected_dashboard == "Execution Time vs. vec_len":
        st.subheader("Execution Time vs. vec_len")
        # Select encoder for chart (right column) from unique values in filtered data
        encoders2 = sorted(filtered_data["encoder"].unique())
        selected_encoder2 = st.selectbox("Select encoder:", encoders2, key="encoder2")

        # Filter data for the selected encoder (right column)
        encoder2_data = filtered_data[filtered_data["encoder"] == selected_encoder2]

        if not encoder2_data.empty:
            # Line chart for execution time vs. vec_len (right column)
            chart = (
                alt.Chart(encoder2_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("vec_len:Q", title="vec_len"),
                    y=alt.Y("avg_execution_time_ms:Q", title="Average Execution Time (ms)"),
                    tooltip=["vec_len:T", "encoder:N", "avg_execution_time_ms:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available for the selected encoder.")