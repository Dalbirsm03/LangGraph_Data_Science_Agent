from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import SystemMessage , HumanMessage
import re
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field 


class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
class Data_Cleaning_Node:

    class Routes(BaseModel):
        route : Literal["Regenerate","Next"] = Field(description="Decide weather to rewrite query by Regenerate again or Next")

    def __init__(self,llm):
        self.llm = llm
        self.router = self.llm.with_structured_output(self.Routes)
    
    def cleaning_suggestions(self, state: PythonAnalystState) -> dict:
        """Generate cleaning suggestions based on the data and user query.
        
        Args:
            state: PythonAnalystState containing raw data and user question
            
        Returns:
            Dict containing cleaning suggestions
        """
        system_message = """
        You are a data cleaning suggestion agent.

        Your job is to:
        1. Always apply a set of **default best-practice cleaning steps** (listed below).
        2. Carefully analyze the user's question to detect any **custom cleaning intent**.
        3. Suggest **column-specific cleaning** if the user's query implies it (e.g., focusing on dates, cities, prices, etc.).
        4. Return a clear, numbered list of cleaning steps in plain English.

        ⚠️ You must NOT return:
        - Any code
        - JSON
        - Logs
        - Python functions

        Default cleaning steps (always include these unless the user explicitly says not to):
        - Drop columns with more than 40% missing values.
        - Impute missing numeric values with the column mean.
        - Impute missing categorical values with the column mode.
        - Convert columns to appropriate data types.
        - Remove duplicate rows.
        - Drop rows with any remaining missing values.
        - Remove rows with extreme outliers using the IQR method.

        Here is a sample of the dataset:
        {sample_data}

        User Query:
        {user_query}

        📌 Output Format:  
        Return only a clean, numbered list of cleaning steps in plain English. Nothing else.
        """
        sample_parts = []
        for i, df in enumerate(state.raw_data):
            if isinstance(df, pd.DataFrame):
                sample_df = df.head(20)
                sample_str = f"File {i + 1} Sample:\n{sample_df.to_string(index=False)}"
                sample_parts.append(sample_str)

        sample_text = "\n\n".join(sample_parts)

        prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message)
        ])

        prompt = prompt_template.invoke({
        "sample_data": sample_text,
        "user_query": state.question
        })

        result = self.llm.invoke(prompt)
        return {"cleaning_suggestion" : result}
    
    
    def cleaning_code(self, state: PythonAnalystState) -> dict:
        """Generate Python code for data cleaning based on suggestions.
        
        Args:
            state: PythonAnalystState containing cleaning suggestions and raw data
            
        Returns:
            Dict containing generated cleaning code
        """
        prompt = PromptTemplate(
            template="""You are a data cleaning code generation agent.

    Your job is to generate a Python function based on the cleaning steps provided by the user:

    {recommended_steps}

    The function must be valid and executable on a pandas DataFrame.

    ---

    🔧 Requirements:
    - Use the sample below as the structure reference.
    - The function input should be a pandas DataFrame.
    - Return the cleaned DataFrame as `data_cleaned`.
    - Include necessary imports *inside* the function.
    - Wrap code in triple backticks with `python`.

    📄 Sample data:
    {sample_text}
    """,
            input_variables=["recommended_steps", "sample_text"]
        )

        sample_parts = []
        for i, df in enumerate(state["raw_data"]):
            if isinstance(df, pd.DataFrame):
                sample_df = df.head(20)
                sample_parts.append(f"File {i + 1} Sample:\n{sample_df.to_string(index=False)}")

        sample_text = "\n\n".join(sample_parts)

        data_cleaning_agent = prompt | self.llm | PythonOutputParser()
        
        response = data_cleaning_agent.invoke({
            "recommended_steps": state["cleaning_suggestion"],
            "sample_text": sample_text
        })
        return {"cleaning_code":response}
    
    
    def cleaning_executor(self, state: PythonAnalystState) -> dict:
        """Execute cleaning code on each DataFrame in the state.
        
        Args:
            state: PythonAnalystState containing raw data and cleaning code
            
        Returns:
            Dict containing list of cleaned DataFrames
        """
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        cleaned_dfs = []
        code = state["cleaning_code"]

        for i, df in enumerate(state["raw_data"]):
            if not isinstance(df, pd.DataFrame):
                logger.warning("Item %(idx)s in raw_data is not a DataFrame, skipping...", 
                             {"idx": i + 1})
                continue
                
            local_vars = {"df": df.copy()}
            
            try:
                # Execute the cleaning code
                exec(code, {}, local_vars)
                
                # Find and execute the cleaning function
                cleaning_func = None
                for val in local_vars.values():
                    if callable(val):
                        cleaning_func = val
                        break
                        
                if cleaning_func is None:
                    raise ValueError("No callable function found in the cleaning code")
                    
                cleaned = cleaning_func(local_vars["df"])
                if not isinstance(cleaned, pd.DataFrame):
                    raise ValueError("Cleaning function did not return a DataFrame")
                    
                cleaned_dfs.append(cleaned)
                logger.info("Successfully cleaned DataFrame %(idx)s", {"idx": i + 1})
                
            except Exception as e:
                logger.error("Error cleaning DataFrame %(idx)s: %(error)s", 
                           {"idx": i + 1, "error": str(e)})
                logger.info("Using original DataFrame %(idx)s due to cleaning error", 
                          {"idx": i + 1})
                cleaned_dfs.append(df)
            
        return {"cleaned_data": cleaned_dfs}
    

    def check_node(self, state: PythonAnalystState) -> dict:
        """Validate the cleaning results against the original suggestions.
        
        Args:
            state: PythonAnalystState containing cleaned data and original suggestions
            
        Returns:
            Dict containing validation result (is_clean)
        """
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

    📋 Output a checklist report in English.

    🧾 Cleaned Data:
    {cleaned_data}

    🧠 Suggestions:
    {suggestions}
    """,
            input_variables=["cleaned_data", "suggestions"]
        )
        sample_parts = []
        for i, df in enumerate(state["cleaned_data"]):
            if isinstance(df, pd.DataFrame):
                sample_df = df.head(15)
                sample_parts.append(f"File {i + 1} Cleaned Sample:\n{sample_df.to_string(index=False)}")
        sample_text = "\n\n".join(sample_parts)
        prompt_filled = prompt.format(
            cleaned_data=sample_text,
            suggestions=state["cleaning_suggestion"]
        )


        result = self.router.invoke(prompt_filled)
        return {"is_clean" : result.route}
    

    def next_route(self, state: PythonAnalystState) -> str:
        """Determine the next step in the cleaning pipeline.
        
        Args:
            state: PythonAnalystState containing cleaning validation result
            
        Returns:
            String indicating next route ("cleaning_suggestions" or "Next")
        """
        if state["is_clean"].lower() == "Regenerate":
            return "cleaning_suggestions"
        return "Next"
