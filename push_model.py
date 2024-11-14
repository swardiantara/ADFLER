import os
from transformers import AutoTokenizer, BertForTokenClassification

# Define paths
model_path = os.path.join("experiments", "llm-based", "bert-base-cased_15")

# Load your model
model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Save it in Hugging Face's format
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Push model to Hugging Face
model.push_to_hub("swardiantara/ADFLER-bert-base-cased")
tokenizer.push_to_hub("swardiantara/ADFLER-bert-base-cased")
