# dspy_cn/rewards/base.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class RewardInput:
    hate_speech: str
    counter_narrative: str

    # optional
    ground_truth: Optional[str] = None
    knowledge: Optional[str] = None
