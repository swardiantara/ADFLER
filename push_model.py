import os
from transformers import XLNetForTokenClassification, AutoTokenizer

# Define paths
model_path = os.path.join("experiments", "llm-based", "xlnet-base-cased_15")

# Load your model
model = XLNetForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Save it in Hugging Face's format
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Push model to Hugging Face
model.push_to_hub("swardiantara/ADFLER-xlnet-base-cased")
tokenizer.push_to_hub("swardiantara/ADFLER-xlnet-base-cased")
