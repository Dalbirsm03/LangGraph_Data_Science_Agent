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
    
class EDA_Node:

    class Routes(BaseModel):
        route : Literal["Regenerate","Next"] = Field(description="Decide weather to rewrite query by Regenerate again or Next")

def __init__(self,llm):
        self.llm=llm
        self.router=self.llm.with_structured_output(self.routes)

def eda_suggestions(self, state: PythonAnalystState) -> dict:
     """Generate EDA suggestions based on the dataset and user query.

Args:
    state: PythonAnalystState containing raw data and user question

Returns:
    Dict containing EDA suggestions
"""
system_message = """
You are an Exploratory Data Analysis (EDA) suggestion agent.

Your job is to:
1. Always apply a set of *default best-practice EDA steps* (listed below).
2. Carefully analyze the user's question to detect any *custom analysis intent*.
3. Suggest *column-specific or target-aware EDA* if the user's query implies it (e.g., regression, classification, specific variable focus).
4. Return a clear, numbered list of EDA steps in plain English.

⚠ You must NOT return:
- Any code
- JSON
- Logs
- Python functions
- Statistical formulas

Default EDA steps (always include these unless the user explicitly says not to):
- Generate dataset summary: shape, datatypes, and null value counts.
- Show descriptive statistics for all numeric columns.
- Show value counts for all categorical columns.
- Plot distributions for all numeric columns (histograms or KDE).
- Plot bar charts for top categorical columns by frequency.
- Generate correlation matrix for numeric columns.
- Plot boxplots to detect outliers in numeric columns.
- Analyze pairwise correlations using scatterplots or heatmaps.
- Identify high-cardinality categorical columns.
- Identify constant or near-constant columns (low variance).
- Plot missing value heatmap or matrix.
- Flag imbalanced target variable (if applicable).
- Plot feature vs target relationships (if target is present).

Here is a sample of the dataset:
{sample_data}

User Query:
{user_query}

📌 Output Format:  
Return only a clean, numbered list of EDA steps in plain English. Nothing else.
"""
   