# Streamlit dashboard for visualization

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Process Analyzer", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("data/predictions.parquet")

df = load_data()

# --- TITLE ---
st.title("AI-Powered OS Performance Analyzer")
st.write("An advanced machine-learning powered dashboard for comprehensive OS process performance analysis.")

# --- SIDEBAR ---
st.sidebar.header("🧭 Controls")
pid_list = sorted(df["pid"].unique())
selected_pid = st.sidebar.selectbox("Select Process ID", pid_list)

filtered = df[df["pid"] == selected_pid]

# --- PROCESS INFO ---
st.subheader(f"📌 Process Information")
col1, col2 = st.columns(2)
col1.metric("PID", selected_pid)
col2.metric("Total Samples", len(filtered))

st.markdown("---")

# --- CPU GRAPH ---
st.write("### 🔵 CPU Usage (with trend)")
fig_cpu = px.line(filtered, x="timestamp", y="cpu_percent",
                  title="CPU Usage Timeline", markers=True)
st.plotly_chart(fig_cpu, use_container_width=True)

# --- MEMORY GRAPH ---
st.write("### 🟣 Memory RSS Growth")
fig_mem = px.line(filtered, x="timestamp", y="mem_rss",
                  title="Memory Usage Timeline", markers=True)
st.plotly_chart(fig_mem, use_container_width=True)

# --- ANOMALY TABLE ---
st.write("### 🚨 Detected Anomalies")
anoms = filtered[filtered["anomaly"] == 1]

if len(anoms):
    st.error(f"⚠ {len(anoms)} anomalies detected for this process!")
    st.dataframe(anoms)
else:
    st.success("🎉 No anomalies detected. System stable!")

# --- FULL DATA ---
st.write("### 📄 Full Dataset")
st.dataframe(filtered)
