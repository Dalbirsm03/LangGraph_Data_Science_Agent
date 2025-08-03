from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser,JsonOutputParser
from langchain_core.messages import SystemMessage , HumanMessage
import re
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field 


def dynamic_sample(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    frac = 1.0 if n < 10_000 else 0.1 if n < 100_000 else 0.03 if n < 1_000_000 else 0.01
    return df.sample(frac=frac, random_state=42)

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
class EDA_Node:

    def __init__(self,llm):
            self.llm=llm

    def eda_suggestions(self, state: PythonAnalystState) -> dict:
        
        prompt = PromptTemplate.from_template(template="""
    You are a senior data analyst working with a cleaned dataset.{cleaned_data}
    User question: "{user_query}"
    ---
    🔍 Your job is to perform *comprehensive exploratory data analysis (EDA)* on the dataset and provide structured insights.

    ### Phase 1: General EDA ({cleaned_data})
    - Shape: rows × columns
    - Data types by column
    - Null values per column
    - Numeric stats: mean, median, min, max, std
    - Skewness & outliers (if any)
    - Categorical: unique values, top 3 frequent
    - Correlation between numeric variables
    - Temporal trends (if date/year present)
    - Anomalies, duplicates, unusual patterns

    Summarize as clean markdown bullet points.
    ---

    ### Phase 2: Query-Focused EDA
    Now focus on the user’s intent: "{user_query}"
    From the above EDA insights, dive deeper and answer:
    - Which features directly relate to the user query?
    - Are there any segments, patterns, trends, or groupings that impact the user’s concern?
    - Provide reasons or hypotheses , data gaps explaining what the user might be noticing or trying to solve.
    No fluff, only relevant insightsKeep it intelligent, not verbose.

    ---
    ### Phase 3: EDA Recheck Suggestions (Optional)
    {eda_recheck_suggestions}
    Only act on this section if suggestions exist. Otherwise, skip.
    ---
    Return three clearly labeled sections:

    #### 📊 General EDA  
    #### 🎯 Query-Focused Insights  
    #### 🛠️ EDA Recheck Suggestions (Optional, only if provided)
    """,
    input_variables=["cleaned_data","user_query","eda_recheck_suggestions"])
        
        chain = prompt | self.llm | StrOutputParser()


        response = chain.invoke({"cleaned_data": dynamic_sample(state["cleaned_data"]).to_markdown(index=False),
                                 "user_query":state["question"],
                                 "eda_recheck_suggestions": state.get("eda_recheck_suggestions", "") if not state.get("is_eda_valid", True) else ""})
        
        return{"eda_suggestion" : response}
    

    def eda_checking(self, state : PythonAnalystState):
        prompt = PromptTemplate.from_template(template="""
        You are a senior data analyst auditing this EDA result:
        ---
        📌 **User Question**: "{question}"
        📊 **EDA Result**:
        {eda_result}
        ---
        ✅ **Checklist**:
        - Basic structure (rows, columns, dtypes)
        - Nulls, outliers, and distributions
        - Summary stats for numerics
        - Top categories for categoricals
        - Key correlations or trends
        - Relevant to the user’s query
        ---
        🎯 **Task**:
        Evaluate the EDA critically.
        - Is it complete and useful?
        - Does it help answer the user’s intent?
        - What’s missing, if anything?
        ---
        🧾 **Output (JSON)**:
        {{
        "is_eda_valid": true/false,
        "missing_points": ["...", "..."],
        "reasoning": "Direct, sharp explanation (no fluff)."
        }}
        """,
        input_variables=["question", "eda_result"])

        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({"eda_result":state["eda_result"],
                                 "question":state["question"]})
        return {
                    "is_eda_valid": response["is_eda_valid"],
                    "eda_recheck_suggestions": response
                }
    
    
    def next_route(self, state: PythonAnalystState):
        if state["is_eda_valid"] == True:
            return "rca_suggestions"
        else:
            return "eda_suggestions"