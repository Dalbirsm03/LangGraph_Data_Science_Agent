from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
import pandas as pd
from ydata_profiling import ProfileReport
import os
import uuid

class Report:

    def __init__(self,llm):
        self.llm = llm
    
    def pandas_report(self, state: PythonAnalystState) -> dict:
        cleaned_dfs = state['raw_data']
        report_paths = []

        for i, df in enumerate(cleaned_dfs):
            if not isinstance(df, pd.DataFrame):
                continue

            # Generate report
            profile = ProfileReport(df, title=f"Pandas Profiling Report {i+1}", minimal=True)

            filename = f"profiling_report_{uuid.uuid4().hex[:8]}.html"
            filepath = os.path.join("reports", filename)
            os.makedirs("reports", exist_ok=True)
            profile.to_file(filepath)

            report_paths.append(filepath)

        return {"profiling_reports": report_paths}
