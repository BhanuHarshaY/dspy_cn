import os
import json
import yaml
import pandas as pd
import openai

from dspy_cn.base_llm import configure_base_llm
from dspy_cn.dspy_program import CounterNarrativeProgram
from dspy_cn.evaluator import DSPyRewardEvaluator


from openai import OpenAI
import json

client = OpenAI()

def llm_judge(hate_speech, generated_cn, model="gpt-4o-mini"):
    prompt = f"""
You are an impartial evaluator.

HATE SPEECH:
{hate_speech}

GENERATED COUNTER-NARRATIVE:
{generated_cn}

Evaluate strictly:

PRS:
0 = No politeness
1 = Partial politeness
2 = Fully polite

CCNC:
0 = Off-topic
1 = Partial coherence
2 = Fully coherent

TS:
0 = Toxic
1 = Mild
2 = Fully non-toxic

Return ONLY JSON:
{{
  "PRS": <0-2>,
  "CCNC": <0-2>,
  "TS": <0-2>
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return {"PRS": 0, "CCNC": 0, "TS": 0}



# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    print("========== DEBUG INFO ==========")
    print("Current working directory:", os.getcwd())
    print("================================\n")

    # -----------------------------
    # Load config
    # -----------------------------
    config_path = os.path.join("dspy_cn", "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    input_path = cfg["data"]["predict1_csv"]
    output_path = cfg["data"]["predict_out_csv"]

    print("Input CSV path:", input_path)
    print("Output CSV path:", output_path)

    # -----------------------------
    # Configure LLM
    # -----------------------------
    configure_base_llm(
        cfg["base_llm"]["model_path"],
        max_tokens=cfg["base_llm"]["max_tokens"],
        temperature=cfg["base_llm"]["temperature"],
    )

    # -----------------------------
    # Load DSPy program
    # -----------------------------
    program = CounterNarrativeProgram()

    opt_path = os.path.join("dspy_cn", "optimized_program.json")
    if os.path.exists(opt_path):
        print("Loading optimized program:", opt_path)
        try:
            program.load(opt_path)
        except TypeError:
            program.load(path=opt_path)
    else:
        print("[INFO] Optimized program not found. Using base program.")

    evaluator = DSPyRewardEvaluator(config_path)

    # -----------------------------
    # Load Data
    # -----------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found at {input_path}")

    df = pd.read_csv(input_path)
    print("\nLoaded DataFrame shape:", df.shape)
    print(df.head())

    has_kn = bool(cfg["data"].get("has_knowledge", False))
    kn_col = cfg["data"].get("knowledge_col", "KNOWLEDGE")

    generated = []
    reward_scores = []

    prs_list = []
    ccnc_list = []
    ts_list = []
    llm_total_list = []

    # -----------------------------
    # Generate Predictions
    # -----------------------------
    print("\nStarting prediction loop...")

    for idx, row in df.iterrows():
        print(f"Processing row {idx+1}/{len(df)}")

        hs = str(row["HATE_SPEECH"])
        gt = str(row["COUNTER_NARRATIVE"])  # No ground truth in test

        pred = program(hate_speech=hs).counter_narrative

        kn = ""
        if has_kn and kn_col in df.columns:
            kn = str(row.get(kn_col, ""))

        knowledge_list = [kn] if kn else None

        # -------------------------
        # DSPy reward evaluation
        # -------------------------
        res = evaluator.evaluate_batch(
            [hs],
            [pred],
            [gt],
            knowledge=knowledge_list,
        )

        reward_score = res["final_score"]

        # -------------------------
        # LLM judge evaluation
        # -------------------------
        # judge_scores = llm_judge(hs, pred)

        # prs = judge_scores["PRS"]
        # ccnc = judge_scores["CCNC"]
        # ts = judge_scores["TS"]

        # llm_total = prs + ccnc + ts

        # -------------------------
        # Store results
        # -------------------------
        generated.append(pred)
        reward_scores.append(reward_score)

        prs_list.append(prs)
        ccnc_list.append(ccnc)
        ts_list.append(ts)
        llm_total_list.append(llm_total)

    print("\nGeneration complete.")

    # -----------------------------
    # Attach to DataFrame
    # -----------------------------
    df["GENERATED_CN"] = generated
    df["DSPY_FINAL_SCORE"] = reward_scores

    df["PRS"] = prs_list
    df["CCNC"] = ccnc_list
    df["TS"] = ts_list
    df["LLM_TOTAL_SCORE"] = llm_total_list

    # -----------------------------
    # Save Output
    # -----------------------------
    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    df.to_csv(abs_output_path, index=False)

    print("\n✅ Saved predictions to:", abs_output_path)
    print("=========================================")


if __name__ == "__main__":
    main()
