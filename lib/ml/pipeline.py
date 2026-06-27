import torch
from functools import cache
from transformers import pipeline, Pipeline
from decouple import config

@cache
def get_cached_pipeline(task: str, model_name: str) -> Pipeline:
    """Initializes a pipeline once per task/model combination and caches the result."""
    try:
        pipe = pipeline(task, model=model_name, dtype=torch.float16, local_files_only=False, token=config("HF_TOKEN"))
    except (OSError, ValueError):
        pipe = pipeline(task, model=model_name, dtype=torch.float16, local_files_only=True, token=config("HF_TOKEN"))
    return pipe