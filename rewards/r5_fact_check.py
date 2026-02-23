# dspy_cn/rewards/r5_fact_check.py
import re
from .base import RewardInput

class FaithfulFactChecking:
    _STRONG = re.compile(r"\b(always|never|everyone|no one)\b", re.I)

    def score(self, inp: RewardInput) -> float:
        penalty = 0.0
        if self._STRONG.search(inp.counter_narrative):
            penalty += 0.3
        return max(0.0, 1.0 - penalty)
