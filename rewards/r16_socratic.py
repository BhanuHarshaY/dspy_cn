# dspy_cn/rewards/r16_socratic.py
from .base import RewardInput

class SocraticEngagement:
    def score(self, inp: RewardInput) -> float:
        cn = inp.counter_narrative.lower()
        if "?" not in cn:
            return 0.2
        if any(x in cn for x in ["oh really", "you think", "seriously"]):
            return 0.1
        if any(x in cn for x in ["have you considered", "what evidence", "how would you feel"]):
            return 1.0
        return 0.6
