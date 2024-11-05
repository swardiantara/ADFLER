import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from itertools import chain
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.metrics import classification_report

from src.data_utils import NERDataset


label2id = {'O': 0, 'B-Event': 1, 'I-Event': 2, 'E-Event': 3, 'S-Event': 4, 'B-NonEvent': 5, 'I-NonEvent': 6, 'E-NonEvent': 7, 'S-NonEvent': 8}

class DroneLogNER:
    def __init__(self, model_name='bert-base-cased', num_labels=9, device='cuda'):
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
    def train(self, train_path, val_path, batch_size=16, epochs=5, learning_rate=2e-5):
        # Create datasets
        train_dataset = NERDataset(train_path, self.tokenizer, label_to_id=label2id)
        val_dataset = NERDataset(val_path, self.tokenizer, label_to_id=label2id)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            val_loss, _, _, _ = self.evaluate(val_loader)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

    
    def decode_tokens(self, input_ids: torch.Tensor) -> List[str]:
        """Convert input IDs back to original text tokens"""
        return self.tokenizer.convert_ids_to_tokens(input_ids)


    def evaluate(self, data_loader):
        self.model.eval()
        total_val_loss = 0
        all_tokens = []
        all_preds = []
        all_labels = []
        # label2id = NERDataset(None, self.tokenizer, label_to_id=label2id).label_to_id
        id2label = {v: k for k, v in label2id.items()}
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=2)

                # Convert predictions back to original format
                batch_predictions = self.convert_predictions_to_original_format(
                    predictions.cpu(),
                    batch["word_ids"],
                    batch["original_words"],
                    id2label
                )
                # Collect predictions and true labels
                all_preds.append(batch_predictions)
                all_labels.append(batch["original_labels"])
                all_tokens.append(batch["original_words"])
                # # Convert predictions and labels to list, filtering out padding (-100)
                # for pred, label, input_id in zip(preds, labels, input_ids):
                #     valid_indices = label != -100
                #     all_preds.append(pred[valid_indices].cpu().numpy())
                #     all_labels.append(label[valid_indices].cpu().numpy())
                #     all_tokens.append(self.decode_tokens(input_id[valid_indices].cpu().numpy()))

        # all_preds = [[id2label[idx] for idx in sample] for sample in all_preds]
        # all_labels = [[id2label[idx] for idx in sample] for sample in all_labels]

        return total_val_loss / len(data_loader), all_preds, all_labels, all_tokens
    

    def predict(self, text):
        """Predict NER tags for a single text input"""
        self.model.eval()
        
        # Tokenize input
        tokens = text.split()  # Simple splitting, you might want to use more sophisticated tokenization
        
        # Get the tokenized input
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Convert predictions to labels
        pred_labels = []
        previous_word_idx = None
        word_ids = []
        
        # Get word IDs for each token
        for i, token in enumerate(tokens):
            token_word_ids = []
            tokenized = self.tokenizer.tokenize(token)
            for _ in tokenized:
                token_word_ids.append(i)
            word_ids.extend(token_word_ids)
        
        # Map predictions back to original tokens
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx:
                id2label = NERDataset(None, self.tokenizer).id2label
                pred_labels.append(id2label[predictions[token_idx]])
                previous_word_idx = word_idx
        
        return list(zip(tokens, pred_labels))

    # def predict(self, text):
    #     """Predict NER tags for a single text input"""
    #     self.model.eval()
        
    #     # Tokenize input
    #     tokens = text.split()  # Simple splitting, you might want to use more sophisticated tokenization
    #     encoded = self.tokenizer(
    #         tokens,
    #         is_split_into_words=True,
    #         return_tensors='pt',
    #         padding=True,
    #         truncation=True
    #     ).to(self.device)
        
    #     with torch.no_grad():
    #         outputs = self.model(**encoded)
    #         predictions = torch.argmax(outputs.logits, dim=2)
        
    #     # Convert predictions to labels
    #     pred_labels = []
    #     word_ids = encoded.word_ids()
    #     last_word_idx = None
    #     id2label = DroneLogDataset(None, self.tokenizer).id2label
        
    #     for token_idx, word_idx in enumerate(word_ids[0]):
    #         if word_idx is None or word_idx == last_word_idx:
    #             continue
    #         pred_labels.append(id2label[predictions[0][token_idx].item()])
    #         last_word_idx = word_idx
        
    #     return list(zip(tokens, pred_labels))

    def convert_predictions_to_original_format(
            self,
            batch_predictions: torch.Tensor,
            batch_word_ids: torch.Tensor,
            batch_original_words: List[List[str]],
            id_to_label: Dict[int, str]
        ) -> List[List[str]]:
            """Convert predictions back to original word-level format."""
            batch_labels = []
            
            for predictions, word_ids, original_words in zip(
                batch_predictions, batch_word_ids, batch_original_words
            ):
                word_level_labels = ["O"] * len(original_words)
                current_word_idx = -1
                current_label_votes = {}
                
                # Convert predictions to labels
                pred_labels = [id_to_label[p.item()] for p in predictions]
                
                # Aggregate votes for each word
                for wp_idx, (word_idx, label) in enumerate(zip(word_ids, pred_labels)):
                    if word_idx < 0:  # Skip special tokens and padding
                        continue
                        
                    if word_idx != current_word_idx:
                        # Resolve votes for previous word
                        if current_word_idx >= 0 and current_label_votes:
                            word_level_labels[current_word_idx] = max(
                                current_label_votes.items(),
                                key=lambda x: (x[1], x[0])
                            )[0]
                        current_word_idx = word_idx
                        current_label_votes = {}
                    
                    # Count votes for this label
                    if label not in current_label_votes:
                        current_label_votes[label] = 0
                    current_label_votes[label] += 1
                
                # Handle last word
                if current_label_votes:
                    word_level_labels[current_word_idx] = max(
                        current_label_votes.items(),
                        key=lambda x: (x[1], x[0])
                    )[0]
                
                batch_labels.append(word_level_labels)
            
            return batch_labels