from typing_extensions import Annotated , TypedDict , Literal, Any

class PythonAnalystState(TypedDict):

    question: str
    raw_data: Any

    cleaning_suggestion: str
    cleaning_code: str
    cleaned_data: Any
    
    is_clean: bool
    eda_code: str
    eda_result: str
    is_eda_done: bool

    visual_plan: str
    visual_code: str
    visual_output: Any
    is_visual_ok: bool