import os
from typing import Any, List
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

            final_answer = None
            try:
                for step in self.graph.stream(state, stream_mode="values"):
                    if "answer" in step:
                        final_answer = step["answer"]
            except Exception as e:
                st.error(f"❌ Error in pipeline execution: {e}")
                return

            if final_answer:
                with st.chat_message("assistant"):
                    st.markdown("🧠 **Final Answer:**")
                    st.write(final_answer)
