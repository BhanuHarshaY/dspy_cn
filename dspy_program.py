# import dspy

# class CounterNarrativeProgram(dspy.Module):
#     """
#     HS -> CN generator. DSPy optimizes prompt/program, not weights.
#     """
#     def __init__(self):
#         super().__init__()
#         self.generate = dspy.Predict(
#     "hate_speech -> counter_narrative",
#     instructions=(
#         "You are an expert at responding to hate speech with calm, empathetic, "
#         "and factual counter-narratives that invite reflection."
#     ),
# )


#     def forward(self, hate_speech: str):
#         return self.generate(hate_speech=hate_speech)

import dspy

# 1. Define the Signature with instructions in the docstring
class CounterNarrativeSignature(dspy.Signature):
    """
    You are a thoughtful and respectful responder.

    Generate a counter-narrative that:
    - Responds directly to the harmful or offensive speech.
    - Is polite, calm, and respectful.
    - Avoids toxic or inflammatory language.
    - Encourages understanding and inclusivity.
    
    IMPORTANT:
    - Use varied sentence openings.
    - Do NOT repeatedly start responses with phrases like:
      "It's important that..."
      "It is important to remember..."
      "We must remember..."
    - Vary tone and structure naturally.
    - Avoid template-like repetition.

    IMPORTANT LENGTH CONSTRAINT:
    - The response must be no more than 3 sentences.
    - Keep it concise and impactful.
    - Do NOT exceed 3 sentences under any circumstance.

    The response should feel human, empathetic, and diverse in style.
    """
    hate_speech = dspy.InputField(desc="a string containing hateful or biased content")
    counter_narrative = dspy.OutputField(desc="a calm, empathetic, and factual response")

class CounterNarrativeProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        # 2. Pass the Signature class to Predict
        self.generate = dspy.Predict(CounterNarrativeSignature)

    def forward(self, hate_speech: str):
        return self.generate(hate_speech=hate_speech)