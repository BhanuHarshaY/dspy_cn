# Category weights (from your spec)
CATEGORY_WEIGHTS = {
    "safety": 0.15,
    "tone": 0.20,
    "refutation": 0.20,
    "alignment": 0.20,
    "grounding": 0.10,
    "language": 0.15,
}

# Within-category weights (exact)
WITHIN = {
    "tone": {"R2": 0.35, "R4": 0.30, "R16": 0.35},
    "refutation": {"R8": 0.45, "R9": 0.25, "R10": 0.30},
    "alignment": {"R3": 0.35, "R11": 0.40, "R12": 0.25},
    "grounding": {"R5": 0.55, "R13": 0.45},
    "language": {"R6": 0.25, "R7": 0.20, "R14": 0.35, "R15": 0.20},
}
