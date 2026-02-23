from __future__ import annotations
from typing import Dict, Any, List, Optional
import yaml

from dspy_cn.weights import CATEGORY_WEIGHTS, WITHIN
from dspy_cn.utils import clamp01

from dspy_cn.rewards import (
    RewardInput,
    SafetyNonToxicity,
    EmpathyReward,
    InputOutputSemanticGrounding,
    NonConfrontationalTone,
    FaithfulFactChecking,
    LexicalDiversity,
    SemanticDiversity,
    MNLIContradiction,
    MNLINonEntailment,
    StanceOpposition,
    AlignWithGTBERTScore,
    AlignWithGTCosine,
    KnowledgeUtilization,
    FluencyPerplexity,
    LengthAppropriateness,
    SocraticEngagement,
)


class DSPyRewardEvaluator:
    def __init__(self, config_path: str | None = None):
        # config_path kept for backward compatibility (not used)
        self.r1 = SafetyNonToxicity()
        self.r2 = EmpathyReward()
        self.r3 = InputOutputSemanticGrounding()
        self.r4 = NonConfrontationalTone()
        self.r5 = FaithfulFactChecking()
        self.r6 = LexicalDiversity()
        self.r7 = SemanticDiversity()
        self.r8 = MNLIContradiction()
        self.r9 = MNLINonEntailment()
        self.r10 = StanceOpposition()
        self.r11 = AlignWithGTBERTScore()
        self.r12 = AlignWithGTCosine()
        self.r13 = KnowledgeUtilization()
        self.r14 = FluencyPerplexity()
        self.r15 = LengthAppropriateness()
        self.r16 = SocraticEngagement()


    def evaluate_batch(
    self,
    hs: List[str],
    cn: List[str],
    gt: Optional[List[str]] = None,
    knowledge: Optional[List[str]] = None,  # Changed from List[List[str]]
) -> Dict[str, Any]:

        n = len(cn)
        gt = gt or [None] * n
        knowledge = knowledge or [None] * n

        # Rest of the code remains the same
        scores = {f"R{i}": [] for i in range(1, 17)}

        for i in range(n):
            inp = RewardInput(
                hate_speech=hs[i],
                counter_narrative=cn[i],
                ground_truth=gt[i],
                knowledge=knowledge[i],
            )
            
            scores["R1"].append(self.r1.score(inp))
            scores["R2"].append(self.r2.score(inp))
            scores["R3"].append(self.r3.score(inp))
            scores["R4"].append(self.r4.score(inp))
            scores["R5"].append(self.r5.score(inp))
            scores["R6"].append(self.r6.score(inp))
            scores["R7"].append(self.r7.score(inp))
            scores["R8"].append(self.r8.score(inp))
            scores["R9"].append(self.r9.score(inp))
            scores["R10"].append(self.r10.score(inp))
            scores["R11"].append(self.r11.score(inp))
            scores["R12"].append(self.r12.score(inp))
            scores["R13"].append(self.r13.score(inp))
            scores["R14"].append(self.r14.score(inp))
            scores["R15"].append(self.r15.score(inp))
            scores["R16"].append(self.r16.score(inp))

        # ---- category aggregation ----
        safety_score = avg(scores["R1"])

        tone_score = (
            WITHIN["tone"]["R2"] * avg(scores["R2"])
            + WITHIN["tone"]["R4"] * avg(scores["R4"])
            + WITHIN["tone"]["R16"] * avg(scores["R16"])
        )

        refutation_score = (
            WITHIN["refutation"]["R8"] * avg(scores["R8"])
            + WITHIN["refutation"]["R9"] * avg(scores["R9"])
            + WITHIN["refutation"]["R10"] * avg(scores["R10"])
        )

        alignment_score = (
            WITHIN["alignment"]["R3"] * avg(scores["R3"])
            + WITHIN["alignment"]["R11"] * avg(scores["R11"])
            + WITHIN["alignment"]["R12"] * avg(scores["R12"])
        )

        grounding_score = (
            WITHIN["grounding"]["R5"] * avg(scores["R5"])
            + WITHIN["grounding"]["R13"] * avg(scores["R13"])
        )

        language_score = (
            WITHIN["language"]["R6"] * avg(scores["R6"])
            + WITHIN["language"]["R7"] * avg(scores["R7"])
            + WITHIN["language"]["R14"] * avg(scores["R14"])
            + WITHIN["language"]["R15"] * avg(scores["R15"])
        )

        final_score = (
            CATEGORY_WEIGHTS["safety"] * safety_score
            + CATEGORY_WEIGHTS["tone"] * tone_score
            + CATEGORY_WEIGHTS["refutation"] * refutation_score
            + CATEGORY_WEIGHTS["alignment"] * alignment_score
            + CATEGORY_WEIGHTS["grounding"] * grounding_score
            + CATEGORY_WEIGHTS["language"] * language_score
        )

        return {
            "final_score": clamp01(final_score),
            "breakdown": {k: avg(v) for k, v in scores.items()},
        }


def avg(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0
