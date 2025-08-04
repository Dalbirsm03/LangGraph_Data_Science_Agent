from typing_extensions import Annotated , TypedDict , Literal, Any,List
import pandas as pd

class PythonAnalystState(TypedDict):

    question: str
    raw_data: List[pd.DataFrame]

    cleaning_suggestion: str
    cleaning_code: str
    cleaned_data: List[pd.DataFrame]
    is_clean: str

    eda_suggestion : str
    eda_code: str
    eda_result: str
    is_eda_valid: bool
    eda_recheck_suggestions : str
    eda_report_path : str
    rca_suggestion: str

    visual_plan: str
    visual_code: str
    visual_output: Any
    is_visual_ok: bool