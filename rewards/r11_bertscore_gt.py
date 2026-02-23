# dspy_cn/rewards/r11_bertscore_gt.py
from bert_score import score
from .base import RewardInput

class AlignWithGTBERTScore:
    def score(self, inp: RewardInput) -> float:
        if not inp.ground_truth:
            return 0.0
        _, _, f1 = score([inp.counter_narrative], [inp.ground_truth], lang="en")
        return float(f1.mean())
