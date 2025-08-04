import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import os
import uuid

# Report generator
def generate_profiling_report(df, save_dir="eda_reports/"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    report_path = os.path.join(save_dir, f"profile_{uuid.uuid4().hex}.html")
    report.to_file(report_path)
    return report_path

# --- Streamlit UI ---
st.title("🔍 EDA with Pandas Profiling")

uploaded_file = st.file_uploader("Upload your CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")

    if st.button("Generate Profiling Report"):
        report_path = generate_profiling_report(df)
        st.session_state["report_path"] = report_path
        st.success("Report generated!")

# Show report if exists
if "report_path" in st.session_state:
    with open(st.session_state["report_path"], "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=800, scrolling=True)

    # Download button
    with open(st.session_state["report_path"], "rb") as f:
        st.download_button("📥 Download Report", f, file_name="profiling_report.html")
