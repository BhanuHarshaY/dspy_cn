# dspy_cn/base_llm.py
import dspy
import os

def configure_base_llm(model, max_tokens, temperature, top_p=0.9):
    lm = dspy.LM(
        model=model,                 # e.g. "gpt-4o-mini"
        provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        # ❌ NO chat=True here
    )

    dspy.settings.configure(lm=lm)
    return lm
