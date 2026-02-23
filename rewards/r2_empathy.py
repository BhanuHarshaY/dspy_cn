from transformers import pipeline
from dspy_cn.utils import clamp01
from .base import RewardInput

class EmpathyReward:
    def __init__(self):
        self.clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )

    def score(self, inp: RewardInput) -> float:
        preds = self.clf(inp.counter_narrative)
        # Handle both flat and nested list formats from the pipeline
        if preds and isinstance(preds[0], list):
            preds = preds[0]  # Unwrap nested list if needed
        
        # preds is now a list of dicts: [{"label": "joy", "score": 0.8}, ...]
        emotion_map = {p["label"]: p["score"] for p in preds}
        empathy = (
            emotion_map.get("joy", 0.0)
            + emotion_map.get("neutral", 0.0)
        ) - (
            emotion_map.get("anger", 0.0)
            + emotion_map.get("disgust", 0.0)
        )
        return float(clamp01(0.5 + empathy))
