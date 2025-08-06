import re
import os
import sys
import uuid
import tempfile
import hashlib
import logging
from typing import List, Any, Dict, Union

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _simple_sns_palette_sanitizer(code: str, default_color: str = "C0") -> str:
    """
    Naive sanitizer: replace sns.*plot(..., palette=...) with color='<default_color>'
    if 'hue' is not present. Ensures color value is quoted.
    """
    # pattern: sns.<word>( ..., palette=..., ... )
    pattern = r"(sns\.\w+\s*\([^)]*?)palette\s*=\s*([^,)\n]+)([,)\n])"
    def repl(match):
        before = match.group(1)
        # ensure quoted color string
        quoted = f"'{default_color}'" if not (default_color.startswith("'") or default_color.startswith('"')) else default_color
        return f"{before}color={quoted}{match.group(3)}"
    try:
        new_code = re.sub(pattern, repl, code, flags=re.S)
        return new_code
    except Exception:
        return code

class Visual_Node:
    def __init__(self, llm) -> None:
        self.llm = llm

    def generate_visual_code(self, state: PythonAnalystState) -> dict:
        """
        Single LLM call -> returns:
          - visual_code: str (no backticks)
          - visual_code_sha1: str
          - visual_plan: str (empty here)
        """
        logger.info("ENTER: generate_visual_code node — state keys: %s", list(state.keys()))
        if "cleaned_data" not in state or not state["cleaned_data"]:
            raise ValueError("cleaned_data not found or empty in state")
        if "question" not in state or not state["question"]:
            raise ValueError("question not found or empty in state")

        column_summary = "\n".join(
            [f"Table {i+1}: {', '.join(df.columns)}" for i, df in enumerate(state["cleaned_data"]) if isinstance(df, pd.DataFrame)]
        )

        unified_prompt = PromptTemplate(
            template=(
                "You are an expert data visualization engineer.\n\n"
                "Produce EXACTLY one runnable Python function wrapped in a python code block and nothing else.\n\n"
                "Function signature must be: def generate_visualizations(df):\n"
                "Place all imports inside the function (pandas, numpy, matplotlib.pyplot as plt). If using seaborn, import seaborn as sns.\n"
                "If you use seaborn `palette=...`, you MUST also include `hue=<column>`. If no hue is needed, use `color='<single-color>'` instead of `palette`.\n"
                "Implement 2-3 purpose-built visuals that directly answer the user's question using only df contents (no hardcoded values).\n"
                "Include titles and axis labels. Call plt.show() after each chart. Do not print or return anything.\n\n"
                "Output EXACT format:\n"
                "```python\n"
                "def generate_visualizations(df):\n"
                "    <code>\n"
                "```\n\n"
                "Inputs:\n"
                "- cleaned_data_columns:\n{column_summary}\n"
                "- user_question:\n{user_question}\n"
                "- eda_summary:\n{eda_result}\n"
                "- rca_summary:\n{rca_result}\n"
            ),
            input_variables=["column_summary", "user_question", "eda_result", "rca_result"]
        )

        eda = state.get("eda_result", "")
        rca = state.get("rca_suggestion", "")

        chain = unified_prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "column_summary": column_summary,
            "user_question": state["question"],
            "eda_result": eda,
            "rca_result": rca
        })

        # extract code block
        code_match = re.search(r"```python\s*(.*?)\s*```", raw, flags=re.S | re.I)
        if code_match:
            visual_code = code_match.group(1).strip()
        else:
            stripped = raw.strip()
            if stripped.startswith("def"):
                visual_code = stripped
            else:
                logger.error("Failed to extract python code block from LLM response. Raw start: %r", raw[:300])
                raise ValueError("Failed to extract python code block from LLM response")

        sha = hashlib.sha1(visual_code.encode("utf-8")).hexdigest()[:8]
        logger.info("EXIT: generate_visual_code — produced visual_code len=%d sha1=%s", len(visual_code), sha)

        # return code and metadata
        return {"visual_plan": "", "visual_code": visual_code, "visual_code_sha1": sha}

    def execute_visual_code(self, state: PythonAnalystState) -> dict:
        """
        Execute the generated `generate_visualizations(df)` and save images.
        Returns:
          {
            "visual_code": str,
            "visual_images": List[Union[{"table":n,"path":...}, {"table":n,"error":...}]],
          }
        """
        logger.info("ENTER: execute_visual_code node — state keys: %s", list(state.keys()))
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

        code = state.get("visual_code")
        if not code:
            logger.error("No visualization code found in state")
            raise ValueError("No visualization code found in state")

        # sanitize (ensures quoted color)
        code = _simple_sns_palette_sanitizer(code)

        cleaned_dfs = state.get("cleaned_data", [])
        if not cleaned_dfs:
            raise ValueError("No cleaned data found in state")

        image_paths: List[Union[Dict[str, Any], str]] = []

        sys.modules.setdefault("matplotlib", matplotlib)
        sys.modules.setdefault("matplotlib.pyplot", plt)

        for idx, df in enumerate(cleaned_dfs):
            table_id = idx + 1
            if not isinstance(df, pd.DataFrame):
                logger.warning("Item %s in cleaned_data is not a DataFrame. Skipping.", table_id)
                image_paths.append({"table": table_id, "error": "not a DataFrame"})
                continue

            exec_globals: Dict[str, Any] = {"__name__": f"visual_exec_{table_id}", "matplotlib": matplotlib}
            exec_globals["matplotlib.pyplot"] = plt
            exec_locals: Dict[str, Any] = {}

            def save_and_track(*args, **kwargs):
                try:
                    temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.png")
                    plt.savefig(temp_file, bbox_inches="tight", dpi=300)
                    plt.close("all")
                    if not os.path.exists(temp_file):
                        raise IOError("Failed to create image file")
                    image_paths.append({"table": table_id, "path": temp_file})
                    logger.info("Saved visualization to %s", temp_file)
                except Exception as e:
                    logger.error("Failed to save image for table %s: %s", table_id, str(e))
                    image_paths.append({"table": table_id, "error": str(e)})

            original_show = getattr(plt, "show", None)
            try:
                exec(code, exec_globals, exec_locals)

                # find function
                generate_func = None
                if "generate_visualizations" in exec_locals:
                    generate_func = exec_locals["generate_visualizations"]
                elif "generate_visualizations" in exec_globals:
                    generate_func = exec_globals["generate_visualizations"]
                else:
                    for val in list(exec_locals.values()) + list(exec_globals.values()):
                        if callable(val):
                            generate_func = val
                            break

                if generate_func is None:
                    raise ValueError("No callable visualization function found in generated code.")

                plt.show = save_and_track
                generate_func(df.copy())

                # save any leftover figures if user didn't call plt.show
                try:
                    if plt.get_fignums():
                        save_and_track()
                except Exception:
                    pass

            except Exception as e:
                logger.error("Error executing visualization on DataFrame %d: %s", table_id, str(e))
                image_paths.append({"table": table_id, "error": str(e)})
            finally:
                if original_show is not None:
                    plt.show = original_show
                else:
                    if hasattr(plt, "show"):
                        try:
                            delattr(plt, "show")
                        except Exception:
                            pass

        is_ok = all(("path" in item and os.path.exists(item["path"]) ) for item in image_paths if isinstance(item, dict))
        visual_images_normalized: List[Union[str, Dict[str, str]]] = []
        for it in image_paths:
            if isinstance(it, dict) and "path" in it:
                visual_images_normalized.append(it["path"])
            else:
                visual_images_normalized.append(it)

        return { 
            "visual_code": code,
            "visual_images": visual_images_normalized,
        }
