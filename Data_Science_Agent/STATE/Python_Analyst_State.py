from typing_extensions import Annotated , TypedDict , Literal, Any,List
import pandas as pd
from typing import Union

class PythonAnalystState(TypedDict):

    question: str
    raw_data: List[pd.DataFrame]

    cleaning_code: str
    cleaned_data: List[pd.DataFrame]
    cleaning_recheck_suggestions : str
    is_clean: bool

    eda_suggestion : str
    eda_code: str
    eda_result: str
    is_eda_valid: bool
    eda_recheck_suggestions : str
    profiling_reports : str

    rca_suggestion: str

    visual_plan: str
    visual_code: str
    visual_images: List[Union[str, dict]]  
    is_visual_ok: bool
    
    final_result: str  