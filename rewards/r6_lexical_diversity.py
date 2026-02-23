# dspy_cn/rewards/r6_lexical_diversity.py
from .base import RewardInput

class LexicalDiversity:
    def score(self, inp: RewardInput) -> float:
        toks = inp.counter_narrative.split()
        if not toks:
            return 0.0
        return len(set(toks)) / len(toks)
