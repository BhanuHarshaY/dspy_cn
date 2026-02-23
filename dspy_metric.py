# from dspy_cn.evaluator import DSPyRewardEvaluator

# _evaluator = DSPyRewardEvaluator("dspy_cn/config.yaml")

# def cn_metric(example, prediction) -> float:
#     hs = str(example.hate_speech)
#     gt = str(example.ground_truth)
#     cn = str(prediction.counter_narrative)

#     knowledge = getattr(example, "knowledge", "")
#     res = _evaluator.evaluate_batch([hs], [cn], [gt], knowledge=[knowledge])
#     return float(res["final_score"])


from dspy_cn.evaluator import DSPyRewardEvaluator
import traceback

_evaluator = DSPyRewardEvaluator("dspy_cn/config.yaml")

def cn_metric(example, prediction) -> float:
    hs = str(example.hate_speech)
    gt = str(example.ground_truth)
    cn = str(prediction.counter_narrative)

    knowledge = getattr(example, "knowledge", "")
    
    # Fix: knowledge should be Optional[str] not List[List[str]]
    knowledge_list = [knowledge] if knowledge else None
    
    try:
        res = _evaluator.evaluate_batch(
            [hs], 
            [cn], 
            [gt], 
            knowledge=knowledge_list
        )
        return float(res["final_score"])
    except Exception as e:
        print(f"\n[ERROR] in cn_metric:")
        print(f"   Hate speech: {hs[:80]}")  
        print(f"   Error: {e}")
        traceback.print_exc()
        raise