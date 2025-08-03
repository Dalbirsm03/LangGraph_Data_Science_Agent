from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
import pandas as pd
from typing import Dict

def dynamic_sample(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    frac = 1.0 if n < 10_000 else 0.1 if n < 100_000 else 0.03 if n < 1_000_000 else 0.01
    return df.sample(frac=frac, random_state=42)

class RCA_Node:
    def __init__(self, llm):
        self.llm = llm

    def rca_node(self,state:PythonAnalystState):
        prompt = PromptTemplate(
            template="""
        You are a senior data analyst tasked with performing **Root Cause and Recommendation Analysis (RCRA)**.

        ---

        📌 **User Question/Concern**:  
        "{user_query}"

        📊 **EDA Summary**:  
        {eda_suggestion}

        📄 **Cleaned Data Sample** (Markdown Table):  
        {sampled_data}

        ---

        🎯 **Your Task**:

        Step 1: Perform **Root Cause Analysis (RCA)**  
        - Confirm if the user's concern is true using data.
        - Identify the top 3–5 contributing factors behind the issue using trends, correlations, segments, or outliers.
        - Highlight specific segments, time periods, and user groups most affected.
        - Ensure all findings are directly connected to the user's question.

        Step 2: Suggest **Actionable Recommendations**  
        - Propose 2–3 specific, measurable, and realistic strategies to improve the situation (e.g., increase bicycle sales next month).
        - Each recommendation must clearly relate to one or more identified root causes.

        If any essential data is missing or limited, call it out.

        ---

        📤 **Output Format (Markdown)**:

        ### 🧠 Root Cause Summary  
        - Brief overview of the core issue and what’s driving it.

        ### 🔍 Contributing Factors  
        - Bullet list of data-driven causes (3–5)

        ### 📌 Segment/Group Focus  
        - Key timeframes, regions, demographics, or categories involved

        ### ⚠️ Data Limitations  
        - Mention if anything is missing that could affect accuracy

        ### 📈 Actionable Recommendations  
        - Bullet list of strategies to fix/improve the situation next cycle
        """,
            input_variables=["user_query", "eda_suggestion", "sampled_data"])
        
        chain = prompt | self.llm | StrOutputParser()


        response = chain.invoke({"user_query" : state["question"],
                                 "eda_suggestion" : state["eda_suggestion"],
                                 "sampled_data" : dynamic_sample(state["cleaned_data"]).to_mark})
        return {"rca_suggestion" : response}

                                 
        
        

