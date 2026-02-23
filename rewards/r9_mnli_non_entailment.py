# dspy_cn/rewards/r9_mnli_non_entailment.py
from transformers import pipeline
from .base import RewardInput

class MNLINonEntailment:
    def __init__(self):
        self.mnli = pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli")

    def score(self, inp: RewardInput) -> float:
        result = self.mnli(inp.hate_speech, [inp.counter_narrative], multi_label=False)
        # result format: {'sequence': ..., 'labels': [...], 'scores': [...]}
        # Prefer non-entailment (contradiction is preferred over entailment)
        if result["labels"][0] != "ENTAILMENT":
            return 1.0
        return 0.0
