from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser,JsonOutputParser
import re
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import uuid
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PythonOutputParser(BaseOutputParser):
    """Parser for extracting Python code from markdown code blocks."""
    
    def parse(self, text: str) -> str:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
class Visual_Node:
    """Node for handling data visualization tasks."""
    
    def __init__(self, llm) -> None:
        self.llm = llm

    def visual_suggestions(self, state: PythonAnalystState) -> dict:

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
            input_variables=["user_query","cleaned_data" ,"eda_result", "rca_result"]
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
    
    def visual_code(self, state: PythonAnalystState) -> dict:
        if "visual_plan" not in state:
            raise KeyError("Visual plan not found in state")
            
        prompt = PromptTemplate(
            template="""
    You are a Python visualization engineer.

    Your job is to generate clean, executable Python code for the approved visualizations.
    ---
    ### 📦 Input:
    Below is the list of charts you must implement:
    {visual_suggestion}
    ---
    ### 🛠 Instructions:
    - Create a single function: `generate_visualizations(df)`
    - `df` is the cleaned pandas DataFrame (assumed available)
    - Place all required imports inside the function
    - For each visualization in the plan:
    - Use correct chart type
    - Include proper title and axis labels
    - Call `plt.show()` (or `fig.show()` for Plotly) to display each chart
    ---
    ### ⚠ Rules:
    - No print(), no return(), no placeholders
    - No markdown, no comments, no explanation
    - No hardcoded values — use only the `df` content
    - Code must be ready to run immediately
    ---
    ### 📤 Output Format:
    Generate the exact number of charts as in the plan. Wrap your full code like this:
    ```python
    def generate_visualizations(df):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        # import plotly.express as px  # only if required

        # Chart 1
        ...
        plt.show()

        # Chart 2
        ...
        plt.show()

        # Chart N (if applicable)
        ...
        plt.show()
    """,
        input_variables=["visual_suggestion"],
        )

        chain = prompt | self.llm | PythonOutputParser()
        response = chain.invoke({"visual_suggestion": state["visual_plan"]})
        return {"visual_code": response}

    def execute_visual_code(self, state: PythonAnalystState) -> dict:
        code = state.get("visual_code")
        if not code:
            raise ValueError("No visualization code found in state")
            
        cleaned_dfs = state.get("cleaned_data", [])
        if not cleaned_dfs:
            raise ValueError("No cleaned data found in state")
            
        image_paths = []

        for idx, df in enumerate(cleaned_dfs):
            if not isinstance(df, pd.DataFrame):
                logger.warning("Item %(idx)s in cleaned_data is not a DataFrame. Skipping.", 
                             {"idx": idx + 1})
                continue

            local_vars = {"df": df.copy()}
            original_show = plt.show

            def save_and_track():
                """Save current plot to a temporary file and track its path."""
                try:
                    temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.png")
                    plt.savefig(temp_file, bbox_inches="tight", dpi=300)
                    plt.close("all")
                    
                    # Verify the file was created successfully
                    if not os.path.exists(temp_file):
                        raise IOError("Failed to create image file")
                        
                    image_paths.append(temp_file)
                    logger.info("Successfully saved visualization to %(path)s", {"path": temp_file})
                    
                except (IOError, ValueError) as e:
                    logger.error("Failed to save image: %(error)s", {"error": str(e)})
                    image_paths.append({"error": f"Failed to save image: {str(e)}"})
                except Exception as e:
                    logger.error("Unexpected error while saving image: %(error)s", {"error": str(e)})
                    image_paths.append({"error": f"Unexpected error: {str(e)}"})

            try:
                exec(code, {}, local_vars)
                generate_func = next((val for val in local_vars.values() if callable(val)), None)
                if generate_func is None:
                    raise ValueError("No function found in generated code.")

                # Patch show to save plots
                plt.show = save_and_track
                generate_func(df)

            except Exception as e:
                logger.error(f"Error executing visualization on DataFrame {idx + 1}: {e}")
                image_paths.append({"error": str(e)})
            finally:
                plt.show = original_show

        return {"visual_images": image_paths}
