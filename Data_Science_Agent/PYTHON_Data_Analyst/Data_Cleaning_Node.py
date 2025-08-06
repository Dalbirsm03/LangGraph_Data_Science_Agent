from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser,JsonOutputParser,StrOutputParser
import re
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field 
import re
import logging


def dynamic_sample(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    n = len(df)
    frac = 1.0 if n < 10_000 else 0.1 if n < 100_000 else 0.03 if n < 1_000_000 else 0.01
    return df.sample(frac=frac, random_state=random_state)


class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
class Data_Cleaning_Node:


    def __init__(self,llm):
        self.llm = llm
    
    def generate_cleaning_code(self, state: PythonAnalystState) -> dict:

        if "raw_data" not in state or not state["raw_data"]:
            raise ValueError("Raw data not found or empty in state")
        if "question" not in state or not state["question"]:
            raise ValueError("User question not found in state")

        # build sample_text from raw_data
        sample_parts = []
        for i, df in enumerate(state["raw_data"]):
            if isinstance(df, pd.DataFrame):
                sample_df = dynamic_sample(df)
                sample_parts.append(f"File {i+1} Sample:\n{sample_df.to_string(index=False)}")
            else:
                raise ValueError(f"Item {i+1} in raw_data is not a valid DataFrame")
        sample_text = "\n\n".join(sample_parts) if sample_parts else ""

        unified_prompt = PromptTemplate(
            template=("""
                        You are a professional data cleaning agent.

        Your task is to generate a robust and accurate Python function named `clean_data(df)` that performs complete and strict data cleaning on a given pandas DataFrame `df`.
        "Sample data:\n{sample_text}\n\n"
        "User question:\n{user_question}"
        At the beginning of the function, **import all required libraries**:
        - `import pandas as pd`
        - `import numpy as np`

        Follow these exact cleaning steps:

        1. **Drop all rows with any missing values** (NaNs).
        2. **Remove all exact duplicate rows**.
        3. **Standardize column names**:
        - Convert all column names to lowercase.
        - Strip leading/trailing spaces.
        - Replace spaces and special characters with underscores.
        4. **Convert column data types**:
        - Convert columns to appropriate types (e.g., datetime, numeric).
        - Skip conversion silently if not possible.
        5. **Detect and handle outliers**:
        - Use the IQR method to clip outliers in numerical columns.
        6. **Clean string and categorical data**:
        - Strip whitespace.
        - Convert to lowercase.
        - Remove special characters.
        - Normalize inconsistent categorical values (e.g., 'Yes', 'YES', 'yes' → 'yes').
        7. **Reset the index** after all cleaning steps.

        Ensure the code is:
        - Clean, readable, and efficient
        - Accurate and production-grade
        - Fully executable

                "Output EXACT format:\n"
                "```python\n"
                "<function code>\n"
                "```\n\n"
            """),
            input_variables=["sample_text", "user_question"]
        )

        chain = unified_prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "sample_text": sample_text,
            "user_question": state["question"]
        })

        # extract code block
        code_match = re.search(r"```python\s*(.*?)\s*```", raw, flags=re.S | re.I)
        if not code_match:
            # try looser extraction
            if raw.strip().startswith("def"):
                cleaning_code = raw.strip()
            else:
                raise ValueError("Failed to extract python code block from LLM response")
        else:
            cleaning_code = code_match.group(1).strip()
        print("Analysis Cleaning")
        return {"cleaning_code": cleaning_code}


    def execute_cleaning_code(self, state: PythonAnalystState) -> dict:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if "raw_data" not in state or not state["raw_data"]:
            raise ValueError("Raw data not found or empty in state")
        if "cleaning_code" not in state or not state["cleaning_code"]:
            raise ValueError("Cleaning code not found in state")

        code = state["cleaning_code"]
        cleaned_dfs = []

        for i, df in enumerate(state["raw_data"]):
            if not isinstance(df, pd.DataFrame):
                logger.warning("Item %s in raw_data is not a DataFrame, skipping...", i + 1)
                continue

            # Prepare a fresh namespace for exec so imports inside function are available where needed.
            ns = {}
            try:
                exec(code, ns, ns)  # populate ns with function and any helpers
            except Exception as e:
                logger.error("Error executing cleaning code for DataFrame %s: %s", i + 1, str(e))
                logger.info("Appending original DataFrame %s due to exec error", i + 1)
                cleaned_dfs.append(df)
                continue

            # find callable cleaning function
            cleaning_func = None
            for val in ns.values():
                if callable(val):
                    # prefer function named clean_dataframe if present
                    if getattr(val, "__name__", "") == "clean_dataframe":
                        cleaning_func = val
                        break
                    if cleaning_func is None:
                        cleaning_func = val

            if cleaning_func is None:
                logger.error("No callable cleaning function found in executed code for DataFrame %s", i + 1)
                cleaned_dfs.append(df)
                continue

            try:
                cleaned = cleaning_func(df.copy())
                if not isinstance(cleaned, pd.DataFrame):
                    raise ValueError("Cleaning function did not return a pandas DataFrame")
                cleaned_dfs.append(cleaned)
                logger.info("Successfully cleaned DataFrame %s", i + 1)
            except Exception as e:
                logger.error("Error running cleaning function on DataFrame %s: %s", i + 1, str(e))
                logger.info("Appending original DataFrame %s due to runtime error", i + 1)
                cleaned_dfs.append(df)
        print("Cleaned")
        return {"cleaned_data": cleaned_dfs}
    