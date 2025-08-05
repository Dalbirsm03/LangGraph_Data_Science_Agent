from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

from Data_Science_Agent.PYTHON_Data_Analyst.Data_Cleaning_Node import Data_Cleaning_Node
from Data_Science_Agent.PYTHON_Data_Analyst.EDA_Node import EDA_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Python_Profiling_Node import Report
from Data_Science_Agent.PYTHON_Data_Analyst.RCA_Node import RCA_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Visual_Node import Visual_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Output_Node import Output_Node

from langgraph.graph import START , END , StateGraph

class Graph_Builder:

    def __init__(self,llm,langsmith_client=None):
        self.llm = llm
        self.langsmith_client = langsmith_client

    def py_graph(self):
        self.graph_builder = StateGraph(PythonAnalystState)
        self.obj = Data_Cleaning_Node(self.llm)
        self.output = Output_Node(self.llm)
        
        self.graph_builder.add_node("Output",self.output.output_parser)

        self.graph_builder.add_node("Clean_Code_Generator",self.obj.generate_cleaning_code)
        self.graph_builder.add_node("Cleaning_Code_Executor",self.obj.execute_cleaning_code)
        self.graph_builder.add_node("Cleaning_Check",self.obj.check_node)

        self.graph_builder.add_edge(START,"Clean_Code_Generator")
        self.graph_builder.add_edge("Clean_Code_Generator","Cleaning_Code_Executor")
        self.graph_builder.add_edge("Cleaning_Code_Executor","Cleaning_Check")
        self.graph_builder.add_conditional_edges("Cleaning_Check",self.obj.next_route,{True:"EDA_Analysis",False:"Clean_Code_Generator"})

        self.obj = EDA_Node(self.llm)
        self.graph_builder.add_node("EDA_Analysis",self.obj.perform_eda_analysis)
        self.graph_builder.add_node("EDA_Code_Executor",self.obj.execute_eda_code)
        self.graph_builder.add_node("EDA_Check",self.obj.eda_checking)

        self.graph_builder.add_edge("EDA_Analysis","EDA_Code_Executor")
        self.graph_builder.add_edge("EDA_Code_Executor","EDA_Check")
        self.graph_builder.add_conditional_edges("EDA_Check",self.obj.next_route,{True:"RCA_Node",False:"EDA_Analysis"})

        self.obj = RCA_Node(self.llm)
        self.graph_builder.add_node("RCA_Node",self.obj.rca_node)
        self.graph_builder.add_edge("Eda_Check","RCA_Node")

        self.obj = Visual_Node(self.llm)
        self.graph_builder.add_node("Visual_Suggestions",self.obj.visual_suggestions)
        self.graph_builder.add_node("Visual_Code_Generator",self.obj.visual_code)
        self.graph_builder.add_node("Visual_Code_Executor",self.obj.execute_visual_code)
        
        self.graph_builder.add_edge("RCA_Node","Visual_Suggestions")
        self.graph_builder.add_edge("Visual_Suggestions","Visual_Code_Generator")
        self.graph_builder.add_edge("Visual_Code_Generator","Visual_Code_Executor")
        self.graph_builder.add_edge("Visual_Code_Executor","Output")

        self.graph_builder.add_edge("Output",END)

    def setup_graph(self,usecase : str):
        if usecase == "Data Analyst Agent":
            self.py_graph()
        return self.graph_builder.compile()