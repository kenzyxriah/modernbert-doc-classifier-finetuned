import os
from transformers import Pipeline
from decouple import config
from lib.ml.pipeline import get_cached_pipeline

HF_REPO_ID = "KenzyXriah/modernbert-doc-classifier"

class DocumentClassifier:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        self.is_zero_shot = pipe.task == "zero-shot-classification"

    def predict(self, text: str, label_definitions: dict) -> dict:
        if self.is_zero_shot:
            descriptive_labels = list(label_definitions.values())
            result = self.pipe(text, candidate_labels=descriptive_labels)
            
            winning_description = result["labels"][0]
            final_company_label = next(key for key, val in label_definitions.items() if val == winning_description)
            confidence = float(result["scores"][0])
        else:
            result = self.pipe(text, top_k=1)[0]
            final_company_label = result["label"]
            confidence = float(result["score"])
            
        return {
            "category": final_company_label,
            "confidence": confidence
        }


def get_document_classifier() -> DocumentClassifier:
    """
    Returns a custom supervised text-classifier if the model weights exist on HF Hub.
    Otherwise, falls back to the ModernBERT zero-shot classifier.
    """
    try:
        pipe = get_cached_pipeline("text-classification", HF_REPO_ID)
    except Exception:
        pipe = get_cached_pipeline(
            task="zero-shot-classification", 
            model_name="MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
        )
    return DocumentClassifier(pipe)
