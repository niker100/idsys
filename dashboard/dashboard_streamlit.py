import streamlit as st
import pandas as pd

# Load the data
data = pd.DataFrame({
    "datum": [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"
    ],
    "kategorie": ["A", "B", "A", "C", "B", "A", "C", "A", "B", "C"],
    "region": ["Nord", "Süd", "Nord", "Ost", "Süd", "West", "Ost", "Nord", "West", "Süd"],
    "umsatz": [100, 150, 120, 130, 170, 90, 160, 110, 180, 140]
})

# 🔧 Expand dashboard to full width
st.set_page_config(layout="wide")

# Convert datum to datetime
data["datum"] = pd.to_datetime(data["datum"])

# Streamlit UI
st.title("Umsatz Dashboard")

# Create two columns: controls and chart
col1, col2 = st.columns([1, 3])  # Adjust ratio as needed

with col1:
    selected_kategorie = st.selectbox("Kategorie auswählen:", sorted(data["kategorie"].unique()))
    st.markdown("**Regionen filtern:**")
    regions = sorted(data["region"].unique())
    selected_regions = [region for region in regions if st.checkbox(region, value=True, key=region)]

# Filter data
filtered_data = data[
    (data["kategorie"] == selected_kategorie) &
    (data["region"].isin(selected_regions))
]

# Chart and metrics in right column
with col2:
    st.metric("Gesamtumsatz", f"{filtered_data['umsatz'].sum()} €")
    st.subheader("Umsatz über Zeit")
    chart_data = filtered_data.sort_values("datum")[["datum", "umsatz"]].set_index("datum")
    st.line_chart(chart_data)