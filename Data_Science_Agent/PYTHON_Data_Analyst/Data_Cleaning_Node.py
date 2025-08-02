from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import 
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
    
    def cleaning_suggestions(self , state: PythonAnalystState):

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
    
    
    def cleaning_code(self, state: PythonAnalystState):
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
    
    
    def cleaning_executor(self, state: PythonAnalystState):
        cleaned_dfs = []

        for i, df in enumerate(state["raw_data"]):
            local_vars = {"df": df.copy()}
            code = state["cleaning_code"]

            try:
                    exec(code, {}, local_vars)

                    for val in local_vars.values():
                        if callable(val):
                            cleaned = val(local_vars["df"])
                            cleaned_dfs.append(cleaned)
                            break
            except Exception as e:
                    print(f"Error cleaning file {i + 1}: {e}")
                    cleaned_dfs.append(df)

            
        return {"cleaned_data":cleaned_dfs}
    
    def check_node(self,state:PythonAnalystState):
        routing = self.router.invoke([
            SystemMessage(content="Decide whether to Generate or Execute the SQL based on the following checker output.If  The original query is correct and does not contain any common mistakes. Therefore, the rewritten query is the same as the original query: then go to execute"),
            HumanMessage(content=checked_result)
        ])


        system_message = """
You are a data cleaning validation agent.

Your job is to analyze a given pandas DataFrame and verify whether it has been cleaned according to a list of suggested or required data cleaning steps.

You must evaluate whether the DataFrame satisfies all the following conditions (as applicable):

✅ *Default Cleaning Requirements* (check only if they were part of the suggested steps):
- Columns with more than 40% missing values have been removed.
- Missing numeric values have been imputed using the mean.
- Missing categorical values have been imputed using the mode.
- Data types of each column are appropriate.
- Duplicate rows have been removed.
- Rows with any remaining missing values have been removed.
- Extreme outliers (beyond 3x IQR) have been removed.

📋 *What to Output:*
- A checklist-style report indicating whether each required step has been satisfied.
- Clearly state any *violations* of the cleaning rules.
- If possible, briefly suggest how to fix each violation (but do not generate any code).

🚫 Do NOT:
- Generate Python code.
- Return DataFrames.
- Explain how to write functions.

Your output must be a plain English validation report clearly assessing whether the data meets the cleaning requirements.
"""