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

    def generate_visual_code(self, state: PythonAnalystState) -> dict:
        if not state.get("cleaned_data") or not state.get("question"):
            raise ValueError("Missing cleaned_data or question")

        # Convert cleaned_data to readable column summary
        column_summary = "\n".join(
            [f"Table {i+1}: {', '.join(map(str, df.columns))}"
            for i, df in enumerate(state["cleaned_data"]) if isinstance(df, pd.DataFrame)]
        )

        # Step 1: Suggest visualizations
        suggestion_prompt = PromptTemplate(
            template="""
    You are an elite data visualization strategist. Your task is to design only the most critical visualizations to help answer the user’s original question.
    ---
    ### 🎯 Objective:
    Extract **2 to 3 visualizations** that expose patterns, trends, or anomalies directly tied to the user's query.  
    No fluff. Only purpose-built visuals that add analytical clarity.
    ---
    ### 📥 Inputs:
    - 🧼 Cleaned Data Sample (columns):  
    {cleaned_data}
    - ❓ User Query:  
    "{user_query}"
    - 📊 EDA Summary:  
    {eda_result}
    - 🧠 RCA Summary:  
    {rca_result}
    ---
    ### 📤 Output Instructions:
    For **each visualization**, provide only:
    1. **Chart Title** – Insight-driven
    2. **Chart Type** – (bar, line, boxplot, heatmap, histogram, etc.)
    3. **X-Axis** – Column or logic
    4. **Y-Axis** – Column or logic
    ---
    ### 📝 Output Format:
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
    - DO NOT write code or explain.
    - Every chart must directly help answer the user’s query.
    - Base visuals strictly on the EDA and RCA findings.
    """,
            input_variables=["user_query", "cleaned_data", "eda_result", "rca_result"]
        )

        suggestion_chain = suggestion_prompt | self.llm | StrOutputParser()
        visual_plan = suggestion_chain.invoke({
            "user_query": state["question"],
            "eda_result": state.get("eda_result", ""),
            "rca_result": state.get("rca_suggestion", ""),
            "cleaned_data": column_summary
        })

        code_prompt = PromptTemplate(
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
    - Place all imports inside the function
    - Implement each visualization exactly as specified
    - Add descriptive titles and axis labels
    - Call `plt.show()` (or `fig.show()` for Plotly) after each chart
    ---
    ### ⚠ Rules:
    - No print(), return(), placeholders, markdown, or explanations
    - No hardcoded values — use only df columns
    - Code must run immediately
    ---
    ### 📤 Output Format:
    ```python
    def generate_visualizations(df):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

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
    input_variables=["visual_suggestion"]
    )

        code_chain = code_prompt | self.llm | PythonOutputParser()
        visual_code = code_chain.invoke({"visual_suggestion": visual_plan})

        return {"visual_code": visual_code}
    
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