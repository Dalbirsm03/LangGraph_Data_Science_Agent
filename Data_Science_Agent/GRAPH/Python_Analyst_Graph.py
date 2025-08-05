from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

from Data_Science_Agent.PYTHON_Data_Analyst.Data_Cleaning_Node import Data_Cleaning_Node
from Data_Science_Agent.PYTHON_Data_Analyst.EDA_Node import EDA_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Python_Profiling_Node import ProfileReport
from Data_Science_Agent.PYTHON_Data_Analyst.RCA_Node import RCA_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Visual_Node import Visual_Node

from langgraph.graph import START , END , StateGraph

class Graph_Builder:

    def __init__(self,llm,langsmith_client=None):
        self.llm = llm
        self.langsmith_client = langsmith_client

    def py_graph(self,llm):
        self.graph_builder = StateGraph(PythonAnalystState)
        self.obj = Data_Cleaning_Node(self.llm)
        self.graph_builder.add_node("Cleaning_Suggestions",self.obj.cleaning_suggestions)
        self.graph_builder.add_node("Cleaning_Code_Generator",self.obj.cleaning_code)
        self.graph_builder.add_node("Cleaning_Code_Executor",self.obj.cleaning_executor)
        self.graph_builder.add_node("Cleaning_Check",self.obj.check_node)

        self.graph_builder.add_edge(START,"Cleaning_Suggestions")
        self.graph_builder.add_edge("Cleaning_Suggestions","Cleaning_Code_Generator")
        self.graph_builder.add_edge("Cleaning_Code_Generator","Cleaning_Code_Executor")
        self.graph_builder.add_edge("Cleaning_Code_Executor","Cleaning_Check")
        self.graph_builder.add_conditional_edges("Cleaning_Check",self.obj.next_route,{True:"Eda_Suggestions",False:"Cleaning_Suggestions"})

        self.obj = EDA_Node(self.llm)
        self.graph_builder.add_node("Eda_Suggestions",self.obj.eda_suggestions)
        self.graph_builder.add_node("Eda_Code_Generator",self.obj.eda_code)
        self.graph_builder.add_node("Eda_Code_Executor",self.obj.execute_eda_code)
        self.graph_builder.add_node("Eda_Check",self.obj.eda_checking)

        self.graph_builder.add_edge("Cleaning_Check","Eda_Suggestions")
        self.graph_builder.add_edge("Eda_Suggestions","Eda_Code_Generator")
        self.graph_builder.add_edge("Eda_Code_Generator","Eda_Code_Executor")
        self.graph_builder.add_edge("Eda_Code_Executor","Eda_Check")
        self.graph_builder.add_conditional_edges("Eda_Check",self.obj.next_route,{True:"RCA_Suggestions",False:"Eda_Suggestions"})

       
       