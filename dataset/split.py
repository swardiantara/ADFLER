import pandas as pd
from collections import defaultdict

# Load the CoNLL data from a text file
def load_conll_data(file_path):
    sentences = []
    sentence = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                token, tag = line.split()
                sentence.append((token, tag))
                
    if sentence:
        sentences.append(sentence)  # Add last sentence if the file ends without a newline
    
    return sentences

# Load the dataset
conll_file_path = "path/to/your/conll_data.txt"
sentences = load_conll_data(conll_file_path)

# Extract all entity types for each sentence
def extract_entity_types(sentence):
    entity_types = set(tag.split('-')[-1] for _, tag in sentence if tag != 'O')
    return list(entity_types)  # Convert to list to represent as multilabels

# Prepare a structured dataset
data = []
for sentence in sentences:
    tokens = [token for token, tag in sentence]
    tags = [tag for token, tag in sentence]
    entity_types = extract_entity_types(sentence)  # List of unique entity types
    
    data.append({
        'sentence': sentence,
        'tokens': tokens,
        'tags': tags,
        'entity_types': entity_types  # Multilabels
    })

# Convert to a DataFrame for easier manipulation
df = pd.DataFrame(data)
