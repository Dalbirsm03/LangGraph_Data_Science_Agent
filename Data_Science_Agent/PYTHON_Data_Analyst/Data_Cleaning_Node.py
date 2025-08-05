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
            template=(
                "You are a data cleaning code generator. Produce EXACTLY one runnable Python function"
                " wrapped in triple backticks with language `python`.\n\n"
                "Requirements:\n"
                "- Function signature: def clean_dataframe(df):\n"
                "- Include necessary imports inside the function (must include import pandas as pd and import numpy as np if used).\n"
                "- Implement default steps unless user disables them: drop cols >40% missing; impute numeric missing with mean; "
                "impute categorical missing with mode; convert dtypes; remove duplicates; drop remaining missing rows; remove outliers with IQR.\n"
                "- Also apply any column-specific steps implied by the user question.\n"
                "- Use robust pandas idioms and return the cleaned DataFrame as `data_cleaned`.\n"
                "- Do not output any text besides the code block.\n\n"
                "Output EXACT format:\n"
                "```python\n"
                "<function code>\n"
                "```\n\n"
                "Sample data:\n{sample_text}\n\n"
                "User question:\n{user_question}"
            ),
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

        return {"cleaned_data": cleaned_dfs}
    

    def check_node(self, state: PythonAnalystState) -> dict:
        prompt = PromptTemplate(
            template="""
    You are a data cleaning validation agent.

    Your job is to analyze a given pandas DataFrame and verify whether it has been cleaned according to a list of suggestions or required data cleaning steps.

    You must evaluate whether the DataFrame satisfies all the following conditions (as applicable):
    - Columns with >40% missing values have been removed.
    - Missing numeric values imputed with mean.
    - Missing categorical values imputed with mode.
    - Appropriate column data types.
    - Duplicates removed.
    - Rows with missing values removed.
    - Outliers removed using 3x IQR.

    🧾 **Cleaned Data (Markdown Table)**:
    {cleaned_data}

    ---

    🎯 **Your Task**:
    Critically assess the cleaned data. Based on your analysis, answer the following:

    Return your response ONLY in the following exact JSON format and nothing else:

    ```json
    {{
    "is_clean": true,
    "missing_points": ["<Issue 1>", "<Issue 2>"],
    "reasoning": "One-line explanation."
    }}

    """,
            input_variables=["cleaned_data"]
        )
        sample_parts = []
        for i, df in enumerate(state["cleaned_data"]):
            if isinstance(df, pd.DataFrame):
                sample_df = dynamic_sample(df)
                sample_parts.append(f"File {i + 1} Sample:\n{sample_df.to_string(index=False)}")
        cleaned_table = "\n\n".join(sample_parts)

        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({
            "cleaned_data": cleaned_table,
        })
        
        if not isinstance(response, dict) or "is_clean" not in response:
            raise ValueError("Invalid response format from cleaning validation")
        print(response["is_clean"])
        return {
            "is_clean": response["is_clean"],
            "cleaning_recheck_suggestions": response
        }
    

    def next_route(self, state: PythonAnalystState) -> str:
        if "is_clean" not in state:
            raise ValueError("Cleaning validation result not found in state")
            
        return "eda_suggestions" if state["is_clean"] else "Clean_Code_Generator"
