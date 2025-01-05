import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import os

@dataclass
class Token:
    text: str
    bioes_tag: str
    
    @property
    def entity_type(self) -> str:
        """Extract entity type from BIOES tag"""
        if self.bioes_tag == "O":
            return "O"
        return self.bioes_tag.split("-")[1]
    
    @property
    def bioes_position(self) -> str:
        """Extract BIOES position from tag"""
        if self.bioes_tag == "O":
            return "O"
        return self.bioes_tag.split("-")[0]

class CoNLLReader:
    def read_file(self, file_path: str) -> List[List[Token]]:
        """Read CoNLL file and return list of sentences with tokens"""
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                # Assuming CoNLL format: token BIOES-tag
                parts = line.split()
                if len(parts) >= 2:
                    token = Token(text=parts[0], bioes_tag=parts[1])
                    current_sentence.append(token)
            
            if current_sentence:
                sentences.append(current_sentence)
                
        return sentences

def convert_to_tagged_text(tokens: List[Token]) -> Tuple[str, str]:
    """Convert tokens to input text and target tagged text"""
    input_text = " ".join(token.text for token in tokens)
    
    # Build target text with XML-style tags
    target_text = ""
    current_entity = None
    entity_tokens = []
    
    for i, token in enumerate(tokens):
        if token.bioes_position == "S":
            # Single token entity
            if entity_tokens:
                # Close previous entity if exists
                target_text += f"</{current_entity}>"
                entity_tokens = []
            target_text += f"<{token.entity_type}>{token.text}</{token.entity_type}>"
            current_entity = None
        
        elif token.bioes_position == "B":
            # Beginning of new entity
            if entity_tokens:
                # Close previous entity if exists
                target_text += f"</{current_entity}>"
                entity_tokens = []
            entity_tokens.append(token.text)
            current_entity = token.entity_type
            
        elif token.bioes_position == "I" or token.bioes_position == "E":
            # Inside or end of entity
            entity_tokens.append(token.text)
            if token.bioes_position == "E":
                # End of entity, output accumulated tokens
                target_text += f"<{current_entity}>{' '.join(entity_tokens)}</{current_entity}>"
                entity_tokens = []
                current_entity = None
                
        else:  # "O" tag
            if entity_tokens:
                # Close previous entity if exists
                target_text += f"</{current_entity}>"
                entity_tokens = []
            target_text += token.text + " "
    
    return input_text.strip(), target_text.strip()

class CoNLLDataset(Dataset):
    def __init__(self, sentences: List[List[Token]], tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for sentence_tokens in sentences:
            input_text, target_text = convert_to_tagged_text(sentence_tokens)
            self.examples.append({
                "input": input_text,
                "target": target_text
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        input_encoding = self.tokenizer(
            example["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            example["target"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

def train_model(train_file: str,
                val_file: str = None,
                model_name: str = "t5-base",
                num_epochs: int = 3,
                batch_size: int = 8,
                learning_rate: float = 5e-5):
    
    # Initialize model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens for entity types
    special_tokens = ["<Event>", "</Event>", "<NonEvent>", "</NonEvent>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Read CoNLL data
    reader = CoNLLReader()
    train_sentences = reader.read_file(train_file)
    
    # Create datasets
    train_dataset = CoNLLDataset(train_sentences, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_file:
        val_sentences = reader.read_file(val_file)
        val_dataset = CoNLLDataset(val_sentences, tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
        
        if val_file:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation loss: {avg_val_loss:.4f}")
    
    return model, tokenizer

def convert_to_conll_format(text: str, predictions: str) -> List[str]:
    """Convert model predictions back to CoNLL format"""
    import re
    
    conll_lines = []
    text_tokens = text.split()
    current_pos = 0
    
    # Extract entities with their positions
    entities = []
    pattern = r"<(Event|NonEvent)>(.*?)</\1>"
    
    for match in re.finditer(pattern, predictions):
        entity_type = match.group(1)
        entity_text = match.group(2)
        entity_tokens = entity_text.split()
        
        if len(entity_tokens) == 1:
            # Single token entity
            tag = f"S-{entity_type}"
        else:
            # Multi-token entity
            tags = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(entity_tokens) - 2) + [f"E-{entity_type}"]
            
        entities.append((entity_tokens, tags))
    
    # Generate CoNLL lines
    current_entity_idx = 0
    current_token_idx = 0
    
    for token in text_tokens:
        if current_entity_idx < len(entities):
            entity_tokens, entity_tags = entities[current_entity_idx]
            if token == entity_tokens[current_token_idx]:
                conll_lines.append(f"{token}\t{entity_tags[current_token_idx]}")
                current_token_idx += 1
                if current_token_idx >= len(entity_tokens):
                    current_entity_idx += 1
                    current_token_idx = 0
                continue
        
        conll_lines.append(f"{token}\tO")
    
    return conll_lines

def predict_and_convert_to_conll(text: str, model, tokenizer, max_length: int = 512) -> List[str]:
    """Make predictions and convert them to CoNLL format"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length
        )
    
    # Decode prediction
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Convert to CoNLL format
    return convert_to_conll_format(text, predicted_text)

# Example usage
if __name__ == "__main__":
    # Example of how to use the code with CoNLL files
    train_file = os.path.join('dataset', 'train_conll_data.txt')
    val_file = os.path.join('dataset', 'test_conll_data.txt')
    
    # Train the model
    model, tokenizer = train_model(train_file, val_file)
    
    # Example prediction
    test_text = "Unknown Error, Cannot Takeoff. Contact DJI support."
    predictions = predict_and_convert_to_conll(test_text, model, tokenizer)
    
    # Print predictions in CoNLL format
    for line in predictions:
        print(line)