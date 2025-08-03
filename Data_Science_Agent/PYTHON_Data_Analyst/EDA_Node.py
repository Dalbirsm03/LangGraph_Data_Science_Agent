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
     """Generate EDA suggestions based on the cleaned data and user query. 


     Args:
            state: PythonAnalystState containing cleaned  data and user question
            
        Returns:
            Dict containing EDA suggestions
     
     """                                
     SystemMessage = """
   