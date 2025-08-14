import streamlit as st
import pandas as pd
import altair as alt
import os
from pathlib import Path
import numpy as np
import ast
import base64
from pathlib import Path

# Load data from CSV
benchmark_file = "multi_parameter_benchmark_results.csv"

# Load the data
data = pd.read_csv(benchmark_file)

# Expand dashboard to full width
st.set_page_config(layout="wide", page_title="IDSYS Dashboard", page_icon="🔍")

# Helper function to display local images
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None

# Add a sidebar for navigation
dashboards = ["Welcome to IDSYS", "FP Ratio in k Identification", "Execution Time vs. vec_len", "System Type Comparison", "PMF & Example Explorer"]
selected_dashboard = st.sidebar.radio("Navigation:", dashboards)

# GitHub repo link in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Resources")
st.sidebar.markdown("📊 [GitHub Repository](https://github.com/niker100/idsys)")
st.sidebar.markdown("📄 [Documentation](https://github.com/niker100/idsys/blob/main/docs/Hauptseminar___ID_System_Evaluation__Report.pdf)")

# Streamlit UI
st.title("Identification Systems Dashboard")
st.subheader(selected_dashboard)

pattern_explanations = {
    "random": "Messages are generated with uniformly random byte values (0-255).",
    "incremental": "Each message is a sequence of zeros followed by a incrementing last byte (mod 2^(gf_exp)) (e.g., `[0, 0, ..., 1]`, `[0, 0, ..., 2]`).",
    "repeated_patterns": "Messages consist of a short, repeating byte pattern (e.g., `[255, 0, 255, 0, ...]`, `[170, 187, 170, 187, ...]`).",
    "only_two": "Messages are constructed so only two unique messages exist, one of them is used at the sender and one at the receiver. This is used to test the consistency of tag generation",
    "low_entropy": "There is a limited alphabet `[0,1,2,3]` from which random symbols are picked to generate the messages.",
    "sparse": "Messages are mostly zeros, with few non-zero byte (255) at a different positions in each message.",
}

# Handle Welcome page
if selected_dashboard == "Welcome to IDSYS":
    st.markdown("""

    **Identification Systems Analysis Framework** - A comprehensive toolkit for evaluating identification coding schemes
    
    ## What is this dashboard about?
    
    This project evaluates noiseless identification (ID) systems, a method of goal-oriented communication, with a focus on ID tagging codes. These systems determine if a sender's and receiver's selected messages match by transmitting only a short tag,  which reduces bandwidth at the cost of potential errors.
    Using a modular test framework developed in Python, we analyze various ID coding schemes and additional scenarios like k-Identification and multi-tag transmission, and assess system encoder performance under non-uniform message distributions.
    """)

    st.image("./pictures/ID_flow-Main Graph.drawio.png", 
            caption="Identification System Flow", 
            use_container_width=True)

    st.markdown("""
        ## How It Works
        
        1. **Message Space**: Both sender and receiver operate on a shared, predefined message space, embedded in a Galois Field $\mathbb{F}$. This space contains $q^m$ unique messages, each of length $m$.
        2. **Message Selection**: The sender selects the message it intends to send. Concurrently, the receiver chooses a candidate message, forming a hypothesis about the sender's choice.
        3. **Encoding**: The sender's chosen message is transformed into a longer codeword. This step often uses an FEC code or hash function to improve the system's error resilience.
        4. **Tag Extraction and Cue Formation**: A single symbol, the tag, and its position are extracted from the codeword. Together, these two pieces of information form the cue to be transmitted.
        5. **Transmission and Tag Comparison**: The compact cue is sent over a noiseless channel. The receiver then compares the received tag with one generated from its own message to check for a match.
        """)
    

    st.markdown("""
    ## Systems Evaluated in IDSYS
    
    - **FEC-based**: Reed-Solomon (RSID), Concatenated Reed-Solomon (RS2ID) and Reed-Muller (RMID) provide a theoretical bound on the error probability due to their linearity and Hamming Distance guarantees.
    - **Hash-based**: SHA1 (SHA1ID) and SHA256 (SHA256ID) use cryptographic hash functions to generate tags.
    - **Baseline**: NoCode (RAW) implemented as baseline.
    """)


    st.markdown("""
    ## Key Metrics
    
    - **False Positive Ratio**: Probability of accepting an incorrect message
    - **Code Rate**: Ratio of useful information to total transmitted data
    - **Execution Time**: Processing time for encoding/verification
    - **Entropy Gain**: How well the system transforms message patterns into uniform tag distributions
    """)
    
    st.markdown("""
    
    ## Explore Our Dashboard
    
    Use the navigation sidebar to explore:
    
    1. **FP Ratio** - Compare false positive rates between systems
    2. **Execution Time** - See how performance scales with message length
    3. **System Comparison** - Direct comparison of different identification approaches
    4. **PMF Explorer** - Analyze message/tag probability distributions
    
    This dashboard is part of the [IDSYS framework](https://github.com/niker100/idsys), an open-source toolkit for identification system analysis.
    """)
    
    st.markdown("---")
    st.subheader("Quick Start with IDSYS")
    
    st.code("""
    # Create an identification system
    from idsys import create_id_system, generate_test_messages
    
    # Create a Reed-Solomon ID system
    system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})
    
    # Generate test messages
    messages = generate_test_messages(vec_len=16, gf_exp=8, count=5)
    
    # Encode a message
    message = messages[0]
    tag = system.send(message)
    
    # Verify the message
    is_valid = system.receive(tag, message)
    """, language="python")

# Handle PMF & Example Explorer separately (different layout)
elif selected_dashboard == "PMF & Example Explorer":
    # Load the PMF and examples CSV
    pdf_csv_path = "pdfs_and_examples.csv"
    
    if not pdf_csv_path:
        st.warning("PMF/example data not found. Please run the collision analysis first.")
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
        
        # Parse tag PMFs
        for col in system_cols:
            pdf_df[col] = pdf_df[col].apply(parse_list)
        
        # Controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Controls")
            
            # Select message pattern
            pattern = st.radio("Select Message Pattern:", pdf_df["pattern"].unique(), )
            
            # Select system for tag PMF
            system = st.radio("Select System for Tag PMF:", system_names)
            
            # Display metadata
            st.subheader("Pattern Info")
            pattern_data = pdf_df[pdf_df["pattern"] == pattern].iloc[0]
            
            # Display how many examples are available
            num_examples = len(pattern_data["examples"])
            st.write(f"{num_examples} example(s) available for this pattern.")

            st.markdown("**Formulas:**")
            st.latex(r"H_2(p) = -\log_2 \sum_i p_i^2", help="Rényi-2 entropy (also called collision entropy). It measures the uncertainty of the distribution. Higher is better (more uniform). Maximum value is 8 for a uniform 8-bit distribution.")
            st.latex(r"G_2 = \begin{cases} \frac{H_2(p_{t}) - H_2(p_{m})}{H_2^{gain}} & H_2(p_t) >= H_2(p_m) \\ \frac{H_2(p_{t}) - H_2(p_{m})}{H_2^{loss}} & H_2(p_t) < H_2(p_m) \end{cases}", help=r"Normalized Entropy Gain measures the increase in entropy from the message to the tag distribution, normalized by the maximum gainable, resp. losable entropy. A value of 1.0 indicates the tag distribution is perfectly uniform. The maximum possible Rényi-entropy gain being $H_2^{gain} = \log_2(|\mathcal{X}|) - H_2(p_m)$ and the maximum possible loss $H_2^{loss}=H_2(p_m)$.")
                     
        
        # Display area
        with col2:
            # Get the data for the selected pattern
            row = pdf_df[pdf_df["pattern"] == pattern].iloc[0]
            msg_pdf = row["msg_pdf"]
            examples = row["examples"]
            tag_pdf = row[f"tag_pdf_{system}"]

            # Extract metrics for display
            msg_h2 = row.get('msg_h2', 0)
            tag_h2 = row.get(f'tag_h2_{system}', 0)
            g2 = row.get(f'g2_{system}', 0)
            fp_rate = row.get(f'fp_rate_{system}', 0)

            # 1. Make G_2 the most prominent metric
            st.metric(
                label=f"Normalized Entropy Gain (G₂) for {system}",
                value=f"{g2:.3f}",
                help="Measures the increase in entropy from message to tag, normalized by the maximum possible entropy (8 bits). Higher is better. A value of 1.0 means the tags are perfectly uniform."
            )
            st.markdown("---")
            
            # 2. Display examples on top
            st.subheader(f"Example Messages for '{pattern}' Pattern", help=pattern_explanations.get(pattern, "No description available."))
            
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
            
            # 3. Message and Tag PMFs side by side
            st.markdown("---")
            col_pdf1, col_pdf2 = st.columns(2)
            
            with col_pdf1:
                st.subheader("Message PMF")
                st.metric("Message Rényi-2 Entropy (H₂)", f"{msg_h2:.3f}")
                
                # Create DataFrame for message PMF
                msg_pdf_df = pd.DataFrame({
                    'Symbol': range(256),
                    'Probability': msg_pdf,
                    'Type': 'Message PMF'
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
                st.subheader(f"Tag PMF ({system})")
                st.metric("Tag Rényi-2 Entropy (H₂)", f"{tag_h2:.3f}")
                
                # Create DataFrame for tag PMF
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

            # 4. Detailed metrics at the bottom
            st.markdown("---")
            st.subheader("System Performance")
            mcol1, mcol2 = st.columns(2)

            with mcol1:
                st.metric(
                    label="Empirical False Positive Rate",
                    value=f"{fp_rate:.2e}",
                    help="The measured probability of a random, non-matching message-tag pair being accepted as valid."
                )
            with mcol2:
                st.metric(
                    label="Theoretical Collision Probability",
                    value=f"{2**(-tag_h2):.2e}",
                    help="The theoretical probability of two different messages producing the same tag, calculated from the tag entropy as P_collision = 2^(-H₂)."
                )


# Handle FP Ratio in k Identification dashboard
elif selected_dashboard == "FP Ratio in k Identification":

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
        filtered_data_1 = filtered_data[filtered_data["system_type"] == selected_system_1]
        filtered_data_2 = filtered_data[filtered_data["system_type"] == selected_system_2]
        
        # Select GF exponent
        # Only show GF exponents present in both datasets
        gf_exps_1 = set(filtered_data_1["gf_exp"].unique())
        gf_exps_2 = set(filtered_data_2["gf_exp"].unique())
        gf_exps = sorted(gf_exps_1 & gf_exps_2)
        if gf_exps:
            selected_gf_exp = st.selectbox("Select GF exponent:", gf_exps)
            filtered_data_1 = filtered_data_1[filtered_data_1["gf_exp"] == selected_gf_exp]
            filtered_data_2 = filtered_data_2[filtered_data_2["gf_exp"] == selected_gf_exp]

        # Select message pattern
        patterns = sorted(filtered_data_1["message_pattern"].unique())
        if patterns:
            selected_pattern = st.selectbox("Message pattern:", patterns)
            filtered_data_1 = filtered_data_1[filtered_data_1["message_pattern"] == selected_pattern]
            filtered_data_2 = filtered_data_2[filtered_data_2["message_pattern"] == selected_pattern]

        st.markdown(" ", help=pattern_explanations.get(selected_pattern, "No description available."))

        # Select number of tags
        # Only show num_tags present in both datasets
        num_tags_1 = set(filtered_data_1["num_tags"].unique())
        num_tags_2 = set(filtered_data_2["num_tags"].unique())
        num_tags = sorted(num_tags_1 & num_tags_2)
        if num_tags:
            selected_tag = st.radio(
                "Select number of tags t:",
                num_tags,
                format_func=lambda x: f"Tags: {x}"
            )
            filtered_data_1 = filtered_data_1[filtered_data_1["num_tags"] == selected_tag]
            filtered_data_2 = filtered_data_2[filtered_data_2["num_tags"] == selected_tag]

        # Add theoretical_fp_rate to the chart data if available
        # Assume theoretical_fp_rate is in filtered_data with columns: num_validation_messages, system_type, theoretical_fp_rate
        theory_data = filtered_data_1[["num_validation_messages", "system_type", "theoretical_fp_rate"]].drop_duplicates()
        theory_data = theory_data.rename(columns={"theoretical_fp_rate": "false_positive_rate"})
        theory_data["system_type"] = "Average Error"

        fp_is_zero = [1,1,1]

        # Pivot data for plotting multiple lines based on test_type
        if not filtered_data_1.empty and not filtered_data_2.empty and not theory_data.empty:



            # combine the two filtered datasets
            # If every false positive rate in one set is always zero dont plot it
            if (filtered_data_1["false_positive_rate"] == 0).all() and (filtered_data_2["false_positive_rate"] == 0).all():
                # Both are all zero, show one (to avoid empty plot)
                filtered_data = filtered_data_1
                fp_is_zero = [0, 0, 1]
            elif (filtered_data_1["false_positive_rate"] == 0).all():
                filtered_data = filtered_data_2
                fp_is_zero[0] = 0
            elif (filtered_data_2["false_positive_rate"] == 0).all():
                filtered_data = filtered_data_1
                fp_is_zero[1] = 0
            else:
                filtered_data = pd.concat([filtered_data_1, filtered_data_2])

            filtered_data = filtered_data[filtered_data["false_positive_rate"] > 0]

            pivot_fp_rate = filtered_data.pivot_table(
                index="num_validation_messages",
                columns="system_type",
                values="false_positive_rate"
            ).sort_index()

            chart_data = pivot_fp_rate.reset_index().melt(
                'num_validation_messages',
                var_name='system_type',
                value_name='false_positive_rate'
            )       

            if (theory_data["false_positive_rate"] == 0).all():
                fp_is_zero[2] = 0
            else:
                # Add theoretical data to the chart data
                theory_data = theory_data[theory_data["false_positive_rate"] > 0]
                if fp_is_zero[0] == 0 and fp_is_zero[1] == 0    :
                    chart_data = theory_data
                else:
                    chart_data = pd.concat([theory_data, chart_data], ignore_index=True)
                    


    with col2:
        st.subheader("False Positive Ratio")
        if not filtered_data.empty and 'pivot_fp_rate' in locals():
            # Line chart for false positive rate by vector length
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "num_validation_messages:Q",
                        title="Number of Validated Messages",
                        scale=alt.Scale(type='log', base=2)
                    ),
                    y=alt.Y(
                        "false_positive_rate:Q",
                        title="False Positive Ratio",
                        scale=alt.Scale(type='log', base=10)
                    ),
                    color=alt.Color("system_type:N", legend=alt.Legend(title="System Type", orient="bottom")),
                    tooltip=["num_validation_messages:Q", "system_type:N", "false_positive_rate:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)

            if fp_is_zero[0] == 0:
                st.markdown(
                    f"<span style='font-size:18px; color:red;'>FP is zero for {selected_system_1}</span>",
                    unsafe_allow_html=True
                )

            if fp_is_zero[1] == 0:
                st.markdown(
                    f"<span style='font-size:18px; color:red;'>FP is zero for {selected_system_2}</span>",
                    unsafe_allow_html=True
                )

            if fp_is_zero[2] == 0:
                st.markdown(
                    f"<span style='font-size:18px; color:red;'>FP is zero for Theoretical Bound</span>",
                    unsafe_allow_html=True
                )

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
        if not filtered_data_1.empty and not filtered_data_2.empty:
            col1, col2 = st.columns(2)

            with col1:

                st.subheader(f"ID Code Rate for {selected_system_1}")

                # Display metrics for first system type
                code_rate_subseq = filtered_data_1['code_rate_subsequently'].mean()
                if code_rate_subseq > 0:
                    power_of_ten = int(np.floor(np.log10(code_rate_subseq)))
                    mantissa = code_rate_subseq / (10 ** power_of_ten)
                    st.metric("ID Code Rate Subsequently", f"{mantissa:.2f}e{power_of_ten}", help="The effective code rate for the scenario of sending more tags only if a positive identification is registered.")
                else:
                    st.metric("ID Code Rate Subsequently", f"{code_rate_subseq:.3f}", help="The effective code rate for the scenario of sending more tags only if a positive identification is registered.")

                code_rate_bulk = filtered_data_1['code_rate_bulk'].mean()
                if code_rate_bulk > 0:
                    power_of_ten = int(np.floor(np.log10(code_rate_bulk)))
                    mantissa = code_rate_bulk / (10 ** power_of_ten)
                    st.metric("ID Code Rate Bulk", f"{mantissa:.2f}e{power_of_ten}", help="The effective code rate for the scenario of sending multiple tags regardless of identification.")
                else:
                    st.metric("ID Code Rate Bulk", f"{code_rate_bulk:.3f}", help="The effective code rate for the scenario of sending multiple tags regardless of identification.")

            with col2:

                st.subheader(f"ID Code Rate for {selected_system_2}")

                # Display metrics for second system type
                code_rate_subseq_2 = filtered_data_2['code_rate_subsequently'].mean()
                if code_rate_subseq_2 > 0:
                    power_of_ten = int(np.floor(np.log10(code_rate_subseq_2)))
                    mantissa = code_rate_subseq_2 / (10 ** power_of_ten)
                    st.metric("ID Code Rate Subsequently", f"{mantissa:.2f}e{power_of_ten}")
                else:
                    st.metric("ID Code Rate Subsequently", f"{code_rate_subseq_2:.3f}")

                code_rate_bulk_2 = filtered_data_2['code_rate_bulk'].mean()
                if code_rate_bulk_2 > 0:
                    power_of_ten = int(np.floor(np.log10(code_rate_bulk_2)))
                    mantissa = code_rate_bulk_2 / (10 ** power_of_ten)
                    st.metric("ID Code Rate Bulk", f"{mantissa:.2f}e{power_of_ten}")
                else:
                    st.metric("ID Code Rate Bulk", f"{code_rate_bulk_2:.3f}")

        else:
            st.write("No data available with current filter settings.")

# Handle execution time vs. vector length dashboard
elif selected_dashboard == "Execution Time vs. vec_len":
    # Filter data for execution time


    execution_time_data = data[data["test_type"] == "execution_time"]
    
    # Calculate throughput in MBps (megabytes per second)
    # throughput_msgs_per_sec * vec_len / 1000000
    execution_time_data = execution_time_data.copy()
    execution_time_data["throughput_MBps"] = execution_time_data["throughput_msgs_per_sec"] * execution_time_data["vec_len"] / 1000000

    # Create three columns: controls and chart
    col1, col2, col3 = st.columns([1, 2, 2])  # Adjust ratio as needed

    with col1:
        st.subheader("Controls")

        # Select system type
        system_types = sorted(execution_time_data["system_type"].unique())
        selected_system_1 = st.selectbox("Select system type 1:", system_types)
        selected_system_2 = st.selectbox("Select system type 2:", system_types)

        # Filter data by selected system type
        filtered_data_1 = execution_time_data[execution_time_data["system_type"] == selected_system_1]
        filtered_data_2 = execution_time_data[execution_time_data["system_type"] == selected_system_2]

        # Select GF exponent
        # Only show GF exponents present in both datasets
        gf_exps = sorted(set(filtered_data_1["gf_exp"].unique()) & set(filtered_data_2["gf_exp"].unique()))
        if gf_exps:
            selected_gf_exp = st.radio(
                "Select GF exponent:",
                gf_exps,
                format_func=lambda x: f"GF(2^{x})"
            )
            filtered_data_1 = filtered_data_1[filtered_data_1["gf_exp"] == selected_gf_exp]
            filtered_data_2 = filtered_data_2[filtered_data_2["gf_exp"] == selected_gf_exp]

        # Select number of tags
        # Only show num_tags present in both datasets
        num_tags = sorted(set(filtered_data_1["num_tags"].unique()) & set(filtered_data_2["num_tags"].unique()))
        if num_tags:
            selected_num_tags = st.radio(
                "Number of tags:",
                num_tags,
                format_func=lambda x: f"Tags: {x}"
            )
            filtered_data_1 = filtered_data_1[filtered_data_1["num_tags"] == selected_num_tags]
            filtered_data_2 = filtered_data_2[filtered_data_2["num_tags"] == selected_num_tags]

        # Display metrics
        avg_messages = execution_time_data['num_messages'].mean()
        if avg_messages > 0:
            power_of_ten = int(np.floor(np.log10(avg_messages)))
        else:
            power_of_ten = 0
        st.markdown(
            f"<span style='font-size:14px;'>avg messages: 10<sup>{power_of_ten}</sup></span>",
            unsafe_allow_html=True
        )

        if not filtered_data_1.empty and not filtered_data_2.empty:
            # Combine the two filtered datasets
            filtered_data = pd.concat([filtered_data_1, filtered_data_2])

            # Pivot data for plotting multiple lines based on vec_len
            pivot_execution_time = filtered_data.pivot_table(
                index="vec_len",
                values="avg_execution_time_ms",
                columns="system_type",
                aggfunc="mean"
            )

            pivot_throughput = filtered_data.pivot_table(
                index="vec_len",
                values="throughput_MBps",
                columns="system_type",
                aggfunc="mean"
            )

    with col2:
        st.subheader("Execution Time")
        
        if not pivot_execution_time.empty:
            # Line chart for execution time vs. vec_len
            chart_data = pivot_execution_time.reset_index().melt(
                id_vars=["vec_len"],
                var_name="system_type",
                value_name="avg_execution_time_ms"
            )
            
            # Determine min/max for both axis across all relevant data
            min_vec_len = data["vec_len"].min()
            max_vec_len = data["vec_len"].max()
            min_exec_time = data["avg_execution_time_ms"].min()
            max_exec_time = data["avg_execution_time_ms"].max()

            # Set some padding for better visualization
            x_domain_vec_len = [min_vec_len, max_vec_len*1.1]
            y_domain_execution_time = [min_exec_time * 0.8, max_exec_time * 1.2]

            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "vec_len:Q",
                        title="Vector Length",
                        scale=alt.Scale(type='log', base=2, domain=x_domain_vec_len)
                    ),
                    y=alt.Y(
                        "avg_execution_time_ms:Q",
                        title="Average Execution Time (ms)",
                        scale=alt.Scale(type='log', domain=y_domain_execution_time)
                    ),
                    color=alt.Color("system_type:N", legend=alt.Legend(title="System Type", orient="bottom")),
                    tooltip=["vec_len:Q", "avg_execution_time_ms:Q", "system_type:N", "num_tags:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available for the selected system and parameters.")

    with col3:
        st.subheader("Throughput")

        if not pivot_throughput.empty:

            # Determine min/max for y axis across all relevant data
            min_throughput = execution_time_data["throughput_MBps"].min()
            max_throughput = execution_time_data["throughput_MBps"].max()

            # Set some padding for better visualization
            y_domain_throughput = [min_throughput * 0.8, max_throughput * 1.2]

            # Line chart for throughput vs. vec_len
            chart_data = pivot_throughput.reset_index().melt(
                id_vars=["vec_len"],
                var_name="system_type",
                value_name="throughput_MBps"
            )
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "vec_len:Q",
                        title="Vector Length",
                        scale=alt.Scale(type='log', base=2, domain=x_domain_vec_len)
                    ),
                    y=alt.Y(
                        "throughput_MBps:Q",
                        title="Throughput (MB/s)",
                        scale=alt.Scale(type='log', domain=y_domain_throughput)
                    ),
                    color=alt.Color("system_type:N", legend=alt.Legend(title="System Type", orient="bottom")),
                    tooltip=["vec_len:Q", "throughput_MBps:Q", "system_type:N", "num_tags:Q"]
                )
                .properties(width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No throughput data available.")

# Handle system type comparison dashboard
elif selected_dashboard == "System Type Comparison":
    # Filter data for system type comparison
    execution_time_data = data[data["test_type"] == "execution_time"]

    # Create two columns: controls and chart
    col1, col2 = st.columns([1, 4])  # Adjust ratio as needed

    with col1:
        st.subheader("Controls")

        # Select GF exponent
        gf_exps = sorted(execution_time_data["gf_exp"].unique())
        if gf_exps:
            selected_gf_exp = st.radio(
                "Select GF exponent:",
                gf_exps,
                format_func=lambda x: f"GF(2^{x})"
            )
            filtered_data = execution_time_data[execution_time_data["gf_exp"] == selected_gf_exp]

        # Select number of tags
        num_tags = sorted(filtered_data["num_tags"].unique())
        if num_tags:
            selected_num_tags = st.radio(
                "Number of tags:", 
                num_tags,
                format_func=lambda x: f"Tags: {x}"
            )
            filtered_data = filtered_data[filtered_data["num_tags"] == selected_num_tags]
            
        # Filter to single-tag systems for fair comparison
        filtered_data = filtered_data[filtered_data["num_tags"] == selected_num_tags]

    with col2:

        # Create three columns for comparison charts
        col1, col2 = st.columns(2)

        if not filtered_data.empty:
            # Prepare data for comparison chart - filter for specific vec_len
            vec_lengths = sorted(filtered_data["vec_len"].unique())
            if vec_lengths:
                selected_vec_len = st.select_slider(
                    "Select vector length:",
                    options=vec_lengths
                )
                comparison_data = filtered_data[filtered_data["vec_len"] == selected_vec_len]
            else:
                    st.write("No vector length data available.")

        with col1:
            st.subheader("Execution Time by System Type")
            
            if not filtered_data.empty:
                # Bar chart comparing system types
                # Always show all system types on the x-axis, even if some are missing in the filtered data
                all_system_types = sorted(data["system_type"].unique())

                # Determine min/max for y axis across all relevant data
                min_exec_time = data["avg_execution_time_ms"].min()
                max_exec_time = data["avg_execution_time_ms"].max()
                y_domain_execution_time = [min_exec_time * 0.8, max_exec_time * 1.2]

                chart = (
                    alt.Chart(comparison_data)
                    .mark_circle(
                        size=100,  # Adjust size for better visibility
                    )
                    .encode(
                        x=alt.X(
                            "system_type:N",
                            title="System Type",
                            sort=all_system_types,
                            scale=alt.Scale(domain=all_system_types)
                        ),
                        y=alt.Y(
                            "avg_execution_time_ms:Q",
                            title="Avg Execution Time (ms)",
                            scale=alt.Scale(type='log', domain=y_domain_execution_time)
                        ),
                        color=alt.Color("system_type:N", legend=None),
                        tooltip=["system_type:N", "avg_execution_time_ms:Q", "vec_len:Q"]
                    )
                    .properties(width=700, height=400)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No data available for comparison.")

        with col2:
            st.subheader("Throughput by System Type")
            
            if not filtered_data.empty:
                # Use the same vec_len as middle column
                if 'selected_vec_len' in locals() and 'comparison_data' in locals():
                    # Bar chart comparing throughput
                    all_system_types = sorted(data["system_type"].unique())

                    # Calculate throughput in MBps (megabytes per second)
                    # throughput_msgs_per_sec * vec_len / 1000000
                    comparison_data = comparison_data.copy()
                    comparison_data["throughput_MBps"] = comparison_data["throughput_msgs_per_sec"] * comparison_data["vec_len"] / 1000000

                    chart = (
                        alt.Chart(comparison_data)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "system_type:N",
                                title="System Type",
                                sort=all_system_types,
                                scale=alt.Scale(domain=all_system_types)
                            ),
                            y=alt.Y(
                                "throughput_MBps:Q",
                                title="Throughput (MB/s)",
                            ),
                            color=alt.Color("system_type:N", legend=None),
                            tooltip=["system_type:N", "throughput_MBps:Q", "vec_len:Q"]
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

# Add version info and attribution at the bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
**IDSYS v0.1.0**  
© 2025 | Open Source Project  
[MIT License](https://github.com/niker100/idsys/blob/main/LICENSE)
""")