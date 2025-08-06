from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

class Output_Node:
    
    def __init__(self,llm):
            self.llm = llm
    def output_parser(self,state :PythonAnalystState):
        
        final_summary_prompt = PromptTemplate(
                input_variables=["user_query", "eda_result", "rca_result", "visual_paths"],
                template="""
            You are a senior data analyst tasked with writing a **final insights summary** based on multiple analysis components.

            Use the inputs below to generate a structured, markdown-formatted summary. The output should reflect *analytical thinking*, *data-backed insights*, and *visual+profiling references*.

            ---

            ### 🎯 Objective  
            Summarize the data and analysis relevant to the user's intent, without repeating the user query.

            ---

            ### 📥 Input Components:

            📊 **EDA Result**:  
            {eda_result}

            🧠 **RCA Insights**:  
            {rca_result}

            🖼️ **Visual Paths**:  
            {visual_paths}


            ---

            ### 📌 Output Format (Markdown)
            Write the response using the exact structure below:

            ### 🧾 Dataset Overview  
            Summarize the dataset (rows, columns, type of data, context).

            ---

            ### 🧠 Root Cause Insights  
            Highlight the key reasons or explanations based on analysis. Be direct, structured, and use bullet points.

            ---
            ### 🖼️ Visual Evidence {visual_paths} 
            Present each visual insight using simple labels (e.g., **Image 1**, **Image 2**), followed by a brief caption or description of what the image shows.  
            **Do not** show any file paths or code — just cleanly formatted markdown with numbered image labels and bullet-pointed interpretations.

            Example format:

            - **Image 1**: Distribution of Selling Price shows right-skewed data with high-value outliers.
            - **Image 2**: Scatter plot of Selling Price vs Year reveals weak positive correlation.
            - **Image 3**: Year-wise price trend highlights increased prices for newer models.

            ---

            ### 📑 Profiling Reports  
            List the profiling HTML files and briefly mention what they reveal.

            ---

            ### ✅ Final Takeaway  
            Give a 2–3 sentence summary of the main insight from the entire pipeline.

            ---
            DO NOT return any code, JSON, or repeat the user's question. Just return the markdown summary.
            """
            )
        chain = final_summary_prompt | self.llm | StrOutputParser()

        response = chain.invoke({
                "user_query": state["question"],
                "eda_result": state["eda_result"],
                "rca_result": state["rca_suggestion"],
                "visual_paths": "\n".join(state.get("visual_images", [])),
            })
        return{"final_result":response}
