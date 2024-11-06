import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from itertools import chain
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report

from src.data_utils import NERDataset, DroneLogDataset, sanity_check


label2id = {'O': 0, 'B-Event': 1, 'I-Event': 2, 'E-Event': 3, 'S-Event': 4, 'B-NonEvent': 5, 'I-NonEvent': 6, 'E-NonEvent': 7, 'S-NonEvent': 8, 'PAD': -100}

class DroneLogNER:
    def __init__(self, model_name='bert-base-cased', num_labels=9, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
    def train(self, args, train_path: str, val_path: str):
        # Create datasets
        train_dataset = DroneLogDataset(args, train_path, self.tokenizer, label2id=label2id)
        val_dataset = DroneLogDataset(args, val_path, self.tokenizer, label2id=label2id)
        
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
        
        # Usage example:
        print("Sanity check on train dataset")
        sanity_check(train_dataset, 2)
        print("Sanity check on test dataset")
        sanity_check(val_dataset, 3)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(args.train_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.train_epochs}'):
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
            val_loss, _, _, _ = self.evaluate(args, val_path, label2id)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

    
    def decode_tokens(self, input_ids: torch.Tensor) -> List[str]:
        """Convert input IDs back to original text tokens"""
        return self.tokenizer.convert_ids_to_tokens(input_ids)


    def evaluate(self, args, data_path, label2id):
        val_dataset = DroneLogDataset(args, data_path, self.tokenizer, label2id=label2id)
        val_loader = DataLoader(val_dataset, batch_size=16)
        self.model.eval()
        total_val_loss = 0
        all_tokens = []
        all_preds = []
        all_labels = []
        # label2id = NERDataset(None, self.tokenizer, label2id=label2id).label2id
        id2label = {v: k for k, v in label2id.items()}
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                word_ids = batch['word_ids']
                
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
                # batch_predictions = self.convert_predictions_to_original_format(
                #     predictions.cpu(),
                #     batch["word_ids"],
                #     batch["original_words"],
                #     id2label
                # )
                # Collect predictions and true labels
                # Convert predictions and labels to list, filtering out padding (-100)
                for pred, label, input_id in zip(predictions, labels, input_ids):
                    valid_indices = label != -100
                    aligned_preds = val_dataset.reconstruct_labels_padding(pred[valid_indices].cpu().numpy(), word_ids.tolist())
                    aligned_labels = val_dataset.reconstruct_labels_padding(label[valid_indices].cpu().numpy(), word_ids.tolist())
                    decoded_input = self.decode_tokens(input_id.cpu().numpy())
                    print(f'decoded_input: {decoded_input}')
                    print(f'aligned_preds: {aligned_preds}')
                    print(f'aligned_labels: {aligned_labels}')
                    all_preds.append(aligned_preds)
                    all_labels.append(aligned_labels)
                    all_tokens.append(decoded_input)

        return total_val_loss / len(val_loader), all_preds, all_labels, all_tokens
    

    def predict(self, text):
        """Predict NER tags for a single text input"""
        self.model.eval()
        id2label = {v: k for k, v in label2id.items()}
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
            id2label: Dict[int, str]
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
                pred_labels = [id2label[p.item()] for p in predictions]
                
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