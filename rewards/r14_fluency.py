# dspy_cn/rewards/r14_fluency.py
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from .base import RewardInput

class FluencyPerplexity:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    def score(self, inp: RewardInput) -> float:
        ids = self.tokenizer(inp.counter_narrative, return_tensors="pt")
        loss = self.model(**ids, labels=ids["input_ids"]).loss
        return float(1.0 / (1.0 + torch.exp(loss)))
