import dspy
import yaml
import pandas as pd

from dspy_cn.base_llm import configure_base_llm
from dspy_cn.dspy_program import CounterNarrativeProgram
from dspy_cn.dspy_metric import cn_metric

def load_trainset(cfg):
    df = pd.read_csv(cfg["data"]["train1_csv"])
    has_kn = bool(cfg["data"].get("has_knowledge", False))
    kn_col = cfg["data"].get("knowledge_col", "KNOWLEDGE")

    exs = []
    for _, row in df.iterrows():
        ex = dspy.Example(
    hate_speech=str(row["HATE_SPEECH"]),
    ground_truth=str(row["COUNTER_NARRATIVE"]),
)
        if has_kn and kn_col in df.columns:
            ex.knowledge = str(row[kn_col])
        exs.append(ex.with_inputs("hate_speech"))
    return exs

def main():
    with open("dspy_cn/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    configure_base_llm(
        cfg["base_llm"]["model_path"],
        max_tokens=cfg["base_llm"]["max_tokens"],
        temperature=cfg["base_llm"]["temperature"],
    )

    trainset = load_trainset(cfg)
    program = CounterNarrativeProgram()

    optimizer = dspy.COPRO(
        metric=cn_metric,
        num_trials=3,
        max_bootstrapped_demos=4,
    )

    optimized = optimizer.compile(
    program,
    trainset=trainset,
    eval_kwargs={}
    )

    optimized.save("dspy_cn/optimized_program.json")
    print("✅ Saved optimized DSPy program to dspy_cn/optimized_program.json")

if __name__ == "__main__":
    main()


# def main():
#     with open("dspy_cn/config.yaml", "r") as f:
#         cfg = yaml.safe_load(f)

#     configure_base_llm(
#         cfg["base_llm"]["model_path"],
#         max_tokens=cfg["base_llm"]["max_tokens"],
#         temperature=cfg["base_llm"]["temperature"],
#     )

#     trainset = load_trainset(cfg)
#     program = CounterNarrativeProgram()

#     optimizer = dspy.COPRO(
#         metric=cn_metric,
#         num_trials=3,
#         max_bootstrapped_demos=4,
#     )

#     try:
#         optimized = optimizer.compile(
#             program,
#             trainset=trainset,
#             eval_kwargs={}
#         )

#         optimized.save("dspy_cn/optimized_program.json")
#         print("✅ Saved optimized DSPy program to dspy_cn/optimized_program.json")
#     except Exception as e:
#         print(f"❌ Optimization failed: {e}")
#         print("You can still use the base program for predictions")
#         raise

# if __name__ == "__main__":
#     main()