import pandas as pd
import re
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser,JsonOutputParser
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
        cleaned_data_sample = "\n\n".join([
        f"File {i+1} Sample:\n{dynamic_sample(df).to_markdown(index=False)}"
        for i, df in enumerate(state.get("cleaned_data", []))
        if isinstance(df, pd.DataFrame)
        ]) or "No cleaned data samples available."

        eda_prompt = PromptTemplate(
            template="""
    You are a senior data analyst working on this dataset:
    {cleaned_data}

    User question: "{user_query}"

    Step-by-step Instructions:

1. **Understand the DataFrame**:
    - Determine shape (rows x columns), data types, and column names.
    - Return this structural metadata.

2. **Descriptive Statistics**:
    - Return `.describe()` statistics for numerical and categorical features separately.
    - Include count, mean, std, min, 25%, 50%, 75%, max for numerics.
    - Include unique count, top value, and frequency for categoricals.

3. **Outlier Detection**:
    - Use IQR method to detect number of outliers in each numerical column.
    - Return count of outliers per column.

4. **Cardinality & Categorical Insight**:
    - For categorical columns: return number of unique values per column.
    - Highlight high-cardinality columns (e.g., >50 unique values).

5. **Correlation Analysis**:
    - Return a dictionary of pairwise Pearson correlations between numerical columns.
    - Only include absolute correlation values > 0.5 (strong correlations).

6. **Data Quality Flags**:
    - Check for:
        - Constant columns (single unique value)
        - Columns with mixed data types (if any)
        - Columns with unusual patterns (like negative age, future dates, etc.)

---

🧠 Constraints:

- Do **not** include any visualizations or plotting logic.
- Do **not** print anything to the console.
- You **must** return all outputs in a single dictionary called `eda_results` with clearly labeled keys.

---

🧪 Output Format:

Generate a Python function named `perform_eda(df)` which:
- Takes a pandas DataFrame `df` as input
- Includes all necessary imports (pandas, numpy, etc.)
- Returns a well-structured dictionary called `eda_results`
- The function must be wrapped in triple backticks using ```python
    """,
            input_variables=["cleaned_data", "user_query"]
        )

        chain = eda_prompt | self.llm | PythonOutputParser()

        code = chain.invoke({
            "cleaned_data": cleaned_data_sample,
            "user_query": state.get("question", "")
        })

        logger.info("EDA function generated")
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
    
    # def eda_checking(self, state: PythonAnalystState) -> dict:
    #     if "eda_result" not in state:
    #         raise ValueError("EDA result not found in state")
            
    #     if "question" not in state:
    #         raise ValueError("User question not found in state")
            
    #     prompt = PromptTemplate.from_template(template="""
    #     You are a senior data analyst auditing this EDA result:
    #     ---
    #     📌 **User Question**: "{question}"
    #     📊 **EDA Result**:
    #     {eda_result}
    #     ---
    #     ✅ **Checklist**:
    #     - Basic structure (rows, columns, dtypes)
    #     - Nulls, outliers, and distributions
    #     - Summary stats for numerics
    #     - Top categories for categoricals
    #     - Key correlations or trends
    #     - Relevant to the user’s query
    #     ---
    #     🎯 **Task**:
    #     Evaluate the EDA critically.
    #     - Is it complete and useful?
    #     - Does it help answer the user’s intent?
    #     - What’s missing, if anything?
    #     ---
    #     🧾 **Output (JSON)**:
    #     {{
    #     "is_eda_valid": True / False (Give strictly Boolean value only),
    #     "missing_points": ["...", "..."],
    #     "reasoning": "Direct, sharp explanation (no fluff)."
    #     }}
    #     """,
    #     input_variables=["question", "eda_result"])

    #     chain = prompt | self.llm | JsonOutputParser()
    #     response = chain.invoke({"eda_result":state["eda_result"],
    #                              "question":state["question"]})
    #     return {
    #                 "is_eda_valid": response["is_eda_valid"],
    #                 "eda_recheck_suggestions": response
    #             }
    
    # def next_route(self, state: PythonAnalystState) -> str:
    #     if "is_eda_valid" not in state:
    #         raise ValueError("EDA validation result not found in state")
            
    #     return "rca_suggestions" if state["is_eda_valid"]EDA_Analysisestions"
