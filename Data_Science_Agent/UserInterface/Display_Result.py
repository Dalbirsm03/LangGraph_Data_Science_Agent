import os
from typing import List
import streamlit as st
import pandas as pd
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message, raw_data: List[pd.DataFrame]):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.raw_data = raw_data

    def display_result_on_ui(self):
        if self.usecase == "Data Analyst Agent":
            state = {
                "question": self.user_message,
                "raw_data": self.raw_data
            }

            with st.status("🔄 Processing analysis...", expanded=False) as status:
                final_answer = None
                visual_images = []
                final_result = None

                shown_steps = set()

                try:
                    for step in self.graph.stream(state, stream_mode="values"):
                        if "cleaning_code" in step and "cleaning_code" not in shown_steps:
                            status.write("🧹 Cleaning data...")
                            shown_steps.add("cleaning_code")

                        if "cleaned_data" in step and "cleaned_data" not in shown_steps:
                            status.write("✨ Data cleaned...")
                            shown_steps.add("cleaned_data")

                        if "eda_code" in step and "eda_code" not in shown_steps:
                            status.write("📊 Performing EDA...")
                            shown_steps.add("eda_code")

                        if "eda_result" in step and "eda_result" not in shown_steps:
                            status.write("📈 Processing EDA results...")
                            shown_steps.add("eda_result")

                        if "rca_suggestion" in step and "rca_suggestion" not in shown_steps:
                            status.write("🔍 Analyzing root causes...")
                            shown_steps.add("rca_suggestion")

                        if "answer" in step and "answer" not in shown_steps:
                            final_answer = step["answer"]
                            status.write("📝 Collecting insights...")
                            shown_steps.add("answer")

                        if "visual_images" in step and "visual_images" not in shown_steps:
                            visual_images = step["visual_images"]
                            status.write("🖼️ Loading visualizations...")
                            shown_steps.add("visual_images")

                        if "final_result" in step and "final_result" not in shown_steps:
                            final_result = step["final_result"]
                            status.write("✨ Preparing final summary...")
                            shown_steps.add("final_result")

                    status.update(label="✅ Analysis complete!", state="complete")

                except Exception as e:
                    st.error(f"❌ Error in pipeline execution: {e}")
                    status.update(label="❌ Analysis failed", state="error")
                    return

            # --- Display Final Text Results ---
            with st.container():
                if final_result:
                    with st.chat_message("assistant"):
                        st.markdown(final_result, unsafe_allow_html=True)
                elif final_answer:
                    with st.chat_message("assistant"):
                        st.markdown("🧠 **Analysis Results**")
                        st.markdown(final_answer, unsafe_allow_html=True)

            # --- Display Visuals Vertically with width 750 ---
            if visual_images:
                with st.container():
                    st.markdown("---")
                    st.markdown("### 🖼️ Visual Evidence")
                    for idx, img_path in enumerate(visual_images):
                        if os.path.exists(img_path):
                            with open(img_path, "rb") as img_file:
                                st.image(img_file.read(), caption=f"Image {idx+1}", width=750)
                        else:
                            st.warning(f"⚠️ Image not found: {img_path}")
