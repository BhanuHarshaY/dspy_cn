# dspy_cn/rewards/r8_mnli_contradiction.py
from transformers import pipeline
from .base import RewardInput

class MNLIContradiction:
    def __init__(self):
        self.mnli = pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli")

    def score(self, inp: RewardInput) -> float:
        # Use zero-shot-classification: hypothesis is the counter_narrative, premise is the hate_speech
        result = self.mnli(inp.hate_speech, [inp.counter_narrative], multi_label=False)
        # result format: {'sequence': ..., 'labels': [...], 'scores': [...]}
        return 1.0 if result["labels"][0] == "CONTRADICTION" else 0.0
