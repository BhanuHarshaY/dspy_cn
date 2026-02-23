# dspy_cn/rewards/r4_non_confrontational.py
from transformers import pipeline
from .base import RewardInput

class NonConfrontationalTone:
    def __init__(self):
        self.emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )

    def score(self, inp: RewardInput) -> float:
        all_results = self.emotion(inp.counter_narrative)
        # Handle both flat and nested list formats from the pipeline
        if all_results and isinstance(all_results[0], list):
            scores = all_results[0]  # Unwrap nested list if needed
        else:
            scores = all_results
        
        neg = sum(s["score"] for s in scores if s["label"] in {"anger", "disgust"})
        return float(max(0.0, 1.0 - neg))
