from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser,JsonOutputParser
import re


class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
class EDA_Node:

    def __init__(self,llm):
            self.llm=llm

    def visual_suggetsions(self,state :PythonAnalystState):
        prompt = PromptTemplate(
            template="""
        You are an elite data visualization strategist. Your task is to design only the most critical visualizations to help answer the user’s original question based on data.
        ---
        ### 🎯 Objective:
        From the cleaned data, EDA, and RCA — extract **2 to 3 visualizations** that expose patterns, trends, or anomalies directly tied to the user's query.  
        No fluff. No general charts. Only purpose-built visuals that add analytical clarity.
        ---
        ### 📥 Inputs:
        - 🧼 Cleaned Data Sample (markdown table):  
        {cleaned_data}
        - ❓ User Query:  
        "{user_query}"
        - 📊 EDA Summary:  
        {eda_result}
        - 🧠 RCA Summary:  
        {rca_result}
        ---
        ### 📤 Output Instructions:
        For **each visualization**, provide only the following:
        1. **Chart Title** – Sharp, insight-driven (e.g., “Q4 Drop in Bicycle Sales – Rural Region”)
        2. **Chart Type** – (e.g., bar, line, boxplot, heatmap, histogram, etc.)
        3. **X-Axis** – Column name or logic (e.g., `month`, `region`)
        4. **Y-Axis** – Column name or logic (e.g., `total_sales`, `count`)
        ---
        ### 📝 Output Format (Markdown Only):
        ### 📈 Suggested Visualizations:
        #### 1. [Chart Title]  
        - **Type**: ...  
        - **X**: ...  
        - **Y**: ...  
        #### 2. [Chart Title]  
        - **Type**: ...  
        - **X**: ...  
        - **Y**: ...  
        [Max: 3 visuals]
        ---
        ### ⚠ Rules:
        - DO NOT write code or suggestions for libraries.
        - DO NOT explain why each chart is useful.
        - DO NOT provide “expected insights”.
        - Every chart must **directly help answer the user’s query**.
        - Visuals must be based strictly on the EDA and RCA findings.
        ---
        """,
            input_variables=["user_query","cleaned_data" "eda_result", "rca_result"]
        )
        chain = prompt | self.llm | StrOutputParser()
        column_summary = "\n".join(
        [f"Table {i+1}: {', '.join(df.columns)}" for i, df in enumerate(state["cleaned_data"])]
    )

        response = chain.invoke({"user_query" : state["question"],
                                    "eda_result" : state["eda_result"],
                                    "rca_result" : state['rca_suggestion'],
                                    "cleaned_data" : column_summary})
        return {"visual_plan" : response}
    
    def visual_code(self, state: PythonAnalystState):
        prompt = PromptTemplate(
            template="""
        You are a Python visualization engineer.

Your task is to generate complete Python code for the suggested visualizations using the cleaned DataFrame and EDA context provided.

---

📦 INPUTS:
- df is the cleaned pandas DataFrame (assume it's already available)
- EDA Summary: {eda_result}
- Visualization Plan: {visual_suggestion}

---

🛠 INSTRUCTIONS:
- Generate *a single function* called generate_visualizations(df)
- Include:
  - Necessary imports (inside the function only)
  - One block of code per chart (based on the plan)
  - Clear titles, axis labels, and proper styling
  - Use plt.show() (or fig.show() for Plotly) to display each plot
- Use only the recommended libraries from the plan unless another is clearly more suitable
- Do not return anything. Do not print. Just show charts.

---

⚠ RULES:
- No hardcoded values outside the DataFrame.
- Do *not* suggest or explain — only return Python code.
- Code must be *immediately executable* and contain *no placeholders*.
- You can assume all required libraries are installed.

---

📦 OUTPUT FORMAT:
```python
def generate_visualizations(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    # import plotly.express as px  # Only if a chart needs Plotly

    # Chart 1: Title
    ...
    plt.show()

    # Chart 2: Title
    ...
    plt.show()""",
            input_variables=["recommended_steps"],
        )
        
        chain = prompt | self.llm | PythonOutputParser()

        response = chain.invoke({"recommended_steps": state["visual_plan"]})

        return {"visual_code": response}


   