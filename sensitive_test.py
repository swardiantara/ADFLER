import os

import pandas as pd
from transformers import pipeline

model_path = os.path.join("experiments", "llm-based-aug20", "roberta-base_15")
model = pipeline('ner', model=model_path)

samples = [
    "Tap Fly Flight Ended. Landing Gear Lowered.",
    "Tap Fly Flight Ended Landing Gear Lowered.",
    "Precision Landing. Correcting Landing Position.",
    "Precision Landing Correcting Landing Position.",
    "Propeller Fell Off. Motor idle. Check whether propellers are installed.",
    "Propeller Fell Off Motor idle Check whether propellers are installed.",
    "Return-to-Home Altitude: 98FT.",
]

for sample in samples:
    preds = pd.DataFrame(model(sample))
    print(f"Sample: {sample}")
    print(f"Preds: {preds}")