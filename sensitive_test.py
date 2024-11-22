import os

import pandas as pd
from transformers import pipeline

model_path = os.path.join("experiments", "llm-based-aug20", "roberta-base_15")
model = pipeline('ner', model=model_path)

samples = [
    "Tap Fly Flight Ended. Landing Gear Lowered.",
    "Tap Fly Flight Ended Landing Gear Lowered.",
    "Tap Fly Flight Ended, Landing Gear Lowered.",
    "Tap Fly Flight Ended: Landing Gear Lowered.",
]

for sample in samples:
    preds = pd.DataFrame(model(sample))
    print(f"Sample: {sample}")
    print(f"Preds: {preds}")