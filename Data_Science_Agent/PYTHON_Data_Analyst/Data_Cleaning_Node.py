from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import ChatPromptTemplate
class Data_Cleaning_Node:

    def __init__(self,llm):
        self.llm = llm
    
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
                sample_df = df.head(3)
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