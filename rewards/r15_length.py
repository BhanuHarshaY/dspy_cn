# dspy_cn/rewards/r15_length.py
from .base import RewardInput

class LengthAppropriateness:
    def score(self, inp: RewardInput) -> float:
        n = len(inp.counter_narrative.split())
        if 30 <= n <= 150:
            return 1.0
        if n < 10 or n > 300:
            return 0.0
        return 0.5
