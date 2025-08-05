import pandas as pd
import re
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dynamic_sample(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    n = len(df)
    frac = 1.0 if n < 10_000 else 0.1 if n < 100_000 else 0.03 if n < 1_000_000 else 0.01
    return df.sample(frac=frac, random_state=random_state)

class PythonOutputParser(BaseOutputParser):
    """Extract Python code from markdown blocks."""
    def parse(self, text: str) -> str:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text

class EDA_Node:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def perform_eda_analysis(self, state: PythonAnalystState) -> dict:
        """Generates EDA Python function from cleaned data + user query."""
        eda_prompt = PromptTemplate.from_template(template="""
You are a senior data analyst working on this dataset:
{cleaned_data}

User question: "{user_query}"

Your job:
- Analyze the dataset and the user query.
- Output meaningful EDA steps to describe the data and address the question.
- Focus on key statistics, nulls, outliers, correlations, and unusual patterns.

Now generate a Python function:
- Name it: perform_eda(df)
- It must:
    - Take a pandas DataFrame as input
    - Contain all required imports (like pandas, numpy)
    - Return a dictionary named `eda_results`
    - Avoid any visualizations or print statements
Wrap the full function in triple backticks using ```python
        """,
        input_variables=["cleaned_data", "user_query"])

        chain = eda_prompt | self.llm | PythonOutputParser()

        cleaned_data_sample = "\n\n".join([
            f"File {i+1} Sample:\n{dynamic_sample(df).to_markdown(index=False)}"
            for i, df in enumerate(state["cleaned_data"]) if isinstance(df, pd.DataFrame)
        ])

        code = chain.invoke({
            "cleaned_data": cleaned_data_sample,
            "user_query": state["question"]
        })

        print("EDA function generated")
        return {"eda_code": code}

    def execute_eda_code(self, state: PythonAnalystState) -> dict:
        """Executes the generated EDA code on the raw data."""
        if "eda_code" not in state:
            raise ValueError("Missing EDA code")
        if "raw_data" not in state:
            raise ValueError("Missing raw data")

        eda_outputs = []
        code = state["eda_code"]

        for i, df in enumerate(state["raw_data"]):
            if not isinstance(df, pd.DataFrame):
                self.logger.warning("Item %d in raw_data is not a DataFrame, skipping...", i + 1)
                continue
            try:
                local_vars = {"df": df.copy()}
                exec(code, {"pd": pd}, local_vars)
                eda_func = next((v for v in local_vars.values() if callable(v)), None)
                if not eda_func:
                    raise ValueError("No valid function found in generated code")
                result = eda_func(df.copy())
                if not isinstance(result, dict):
                    raise ValueError("EDA function must return a dictionary")
                eda_outputs.append(result)
                self.logger.info("Successfully executed EDA on DataFrame %d", i + 1)
            except Exception as e:
                self.logger.error("Failed EDA on DataFrame %d: %s", i + 1, str(e))
                eda_outputs.append({"error": str(e)})

        return {"eda_result": eda_outputs}

    

    
    def eda_checking(self, state: PythonAnalystState) -> dict:
        if "eda_result" not in state:
            raise ValueError("EDA result not found in state")
            
        if "question" not in state:
            raise ValueError("User question not found in state")
            
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
        "is_eda_valid": True / False (Give strictly Boolean value only),
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
    
    def next_route(self, state: PythonAnalystState) -> str:
        if "is_eda_valid" not in state:
            raise ValueError("EDA validation result not found in state")
            
        return "RCA_Node" if state["is_eda_valid"] else "Eda_Suggestions"
