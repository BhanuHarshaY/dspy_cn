# dspy_cn/rewards/r7_semantic_diversity.py
from sentence_transformers import SentenceTransformer
from .base import RewardInput

class SemanticDiversity:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, inp: RewardInput) -> float:
        emb = self.model.encode(inp.counter_narrative)
        return float(1.0)  # placeholder (batch handled elsewhere)
