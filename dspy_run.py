import dspy
import yaml

from dspy_cn.base_llm import configure_base_llm
from dspy_cn.dspy_program import CounterNarrativeProgram
from dspy_cn.evaluator import DSPyRewardEvaluator

def main():
    with open("dspy_cn/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    configure_base_llm(
        cfg["base_llm"]["model_path"],
        max_tokens=cfg["base_llm"]["max_tokens"],
        temperature=cfg["base_llm"]["temperature"],
    )

    program = CounterNarrativeProgram.load("dspy_cn/optimized_program.json")
    evaluator = DSPyRewardEvaluator("dspy_cn/config.yaml")

    hs = "Immigrants are ruining the country."
    gt = "People come for many reasons, and blaming a whole group isn’t fair. Can we look at facts and individual actions instead?"
    pred = program(hate_speech=hs).counter_narrative

    res = evaluator.evaluate_batch([hs], [pred], [gt], knowledge=[""])
    print("HS:", hs)
    print("\nCN:", pred)
    print("\nFinal Score:", res["final_score"])
    print("\nBreakdown:", res["breakdown"])

if __name__ == "__main__":
    main()
