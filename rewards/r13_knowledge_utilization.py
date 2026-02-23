# dspy_cn/rewards/r13_knowledge_utilization.py
from sentence_transformers import SentenceTransformer, util
from .base import RewardInput

class KnowledgeUtilization:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, inp: RewardInput) -> float:
        if not inp.knowledge:
            return 0.0
        cn = self.model.encode(inp.counter_narrative, convert_to_tensor=True)
        kn = self.model.encode(inp.knowledge, convert_to_tensor=True)
        return float(util.cos_sim(cn, kn).max())
