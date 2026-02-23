# dspy_cn/rewards/r10_stance.py
from transformers import pipeline
from .base import RewardInput

class StanceOpposition:
    def __init__(self):
        self.clf = pipeline("zero-shot-classification", model="roberta-large-mnli")

    def score(self, inp: RewardInput) -> float:
        result = self.clf(inp.hate_speech, [inp.counter_narrative], multi_label=False)
        # result format: {'sequence': ..., 'labels': [...], 'scores': [...]}
        return 1.0 if result["labels"][0] == "CONTRADICTION" else 0.0
