import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelForTokenClassification
# from seqeval.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple, Set
import logging
from tqdm import tqdm
from collections import defaultdict


class NERDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-cased",
        max_length: int = 128,
        label_pad_token: str = "PAD"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        self.label_pad_token = label_pad_token
        
        # Load and process data
        self.texts, self.labels = self.read_conll(data_path)
        self.label2id = self.create_label_map()
        self.processed_data = self.process_all_data()
        
    def read_conll(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Read CoNLL format file and return texts and labels."""
        texts, labels = [], []
        current_words, current_labels = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    word, label = line.split()
                    current_words.append(word)
                    current_labels.append(label)
                elif current_words:  # Empty line indicates end of sentence
                    texts.append(current_words)
                    labels.append(current_labels)
                    current_words, current_labels = [], []
                    
            # Add the last sentence if file doesn't end with empty line
            if current_words:
                texts.append(current_words)
                labels.append(current_labels)
                
        return texts, labels
    
    def create_label_map(self) -> Dict[str, int]:
        """Create mapping from labels to IDs."""
        unique_labels = set()
        for label_seq in self.labels:
            unique_labels.update(label_seq)
        unique_labels.add(self.label_pad_token)
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    def align_labels_to_tokens(
        self,
        words: List[str],
        labels: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
        """Align original labels to WordPiece tokens."""
        tokens = []
        aligned_labels = []
        word_ids = []  # For backward alignment during evaluation
        
        for word_idx, (word, label) in enumerate(zip(words, labels)):
            # Tokenize word into WordPieces
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
                
            # Add tokens and align labels
            tokens.extend(word_tokens)
            
            # First subword gets the original label
            aligned_labels.append(label)
            word_ids.append(word_idx)
            
            # Additional subwords get PAD label
            for _ in range(len(word_tokens) - 1):
                aligned_labels.append(self.label_pad_token)
                word_ids.append(word_idx)
                
        return tokens, aligned_labels, word_ids
    
    def process_all_data(self) -> List[Dict]:
        """Process all sentences in the dataset."""
        processed_data = []
        
        for words, labels in zip(self.texts, self.labels):
            # Align labels to WordPiece tokens
            tokens, aligned_labels, word_ids = self.align_labels_to_tokens(words, labels)
            
            # Convert tokens to input IDs and attention mask
            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Convert labels to IDs
            label_ids = [self.label2id[label] for label in aligned_labels]
            
            # Pad or truncate labels and word_ids
            if len(label_ids) > self.max_length - 2:  # Account for [CLS] and [SEP]
                label_ids = label_ids[:self.max_length-2]
                word_ids = word_ids[:self.max_length-2]
            else:
                pad_length = self.max_length - 2 - len(label_ids)
                label_ids.extend([self.label2id[self.label_pad_token]] * pad_length)
                word_ids.extend([-1] * pad_length)
            
            # Add labels for [CLS] and [SEP]
            label_ids = [self.label2id[self.label_pad_token]] + label_ids + [self.label2id[self.label_pad_token]]
            word_ids = [-1] + word_ids + [-1]
            
            processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(label_ids),
                'word_ids': torch.tensor(word_ids)
            })
            
        return processed_data
    
    def align_predictions_to_words(
        self,
        predictions: torch.Tensor,
        word_ids: torch.Tensor
    ) -> List[int]:
        """
        Align WordPiece-level predictions back to word-level.
        Uses the first subword's prediction as the word's prediction.
        """
        word_preds = []
        current_word_id = -1
        
        for pred, word_id in zip(predictions.tolist(), word_ids.tolist()):
            if word_id != current_word_id and word_id != -1:
                word_preds.append(pred)
                current_word_id = word_id
                
        return word_preds
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]
    
class NERTrainer:
    def __init__(
        self,
        train_dataset: NERDataset,
        val_dataset: NERDataset,
        model_name: str = "bert-base-cased",
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(train_dataset.label2id)
        ).to(device)
        
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.id2label = {v: k for k, v in train_dataset.label2id.items()}
        
    def train(self):
        best_f1 = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate and save predictions
            metrics, analysis_df = self.evaluate()
            print(f"Metrics for epoch {epoch+1}:")
            print(f"Boundary Detection - F1: {metrics['boundary']['f1']:.4f}")
            print(f"Type Classification - F1: {metrics['type']['f1']:.4f}")
            
            # Save best model
            combined_f1 = metrics['boundary']['f1']
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                torch.save(self.model.state_dict(), os.path.join("development", "best_model.pt"))
                analysis_df.to_excel(os.path.join("development", "error_analysis.xlsx"), index=False)
    
    def decode_bioes(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """Extract entity spans from BIOES labels."""
        spans = []
        start_idx = None
        current_type = None
        
        for i, label in enumerate(labels):
            if label == "O" or label == "PAD":
                if start_idx is not None:
                    start_idx = None
                continue
                
            prefix, entity_type = label.split("-")
            
            if prefix in ["B", "S"]:
                if start_idx is not None:
                    start_idx = None
                start_idx = i
                current_type = entity_type
                
                if prefix == "S":
                    spans.append((i, i, entity_type))
                    start_idx = None
                    
            elif prefix == "E" and start_idx is not None:
                if entity_type == current_type:
                    spans.append((start_idx, i, entity_type))
                start_idx = None
                
            elif prefix == "I":
                if start_idx is None or entity_type != current_type:
                    start_idx = None
        
        return spans
    
    def evaluate(self) -> Tuple[Dict, pd.DataFrame]:
        self.model.eval()
        all_true_spans = []
        all_pred_spans = []
        
        # For error analysis DataFrame
        analysis_records = []
        message_id = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word_ids = batch['word_ids']
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Convert to word-level predictions
                for i in range(len(input_ids)):
                    # Get original tokens for this sample
                    sample_tokens = self.val_loader.dataset.tokenizer.convert_ids_to_tokens(
                        input_ids[i][attention_mask[i] == 1]
                    )
                    
                    # Get word-level predictions and labels
                    word_preds = self.train_loader.dataset.align_predictions_to_words(
                        predictions[i], word_ids[i]
                    )
                    word_labels = self.train_loader.dataset.align_predictions_to_words(
                        batch['labels'][i], word_ids[i]
                    )
                    
                    # Convert IDs to labels
                    pred_labels = [self.id2label[p] for p in word_preds]
                    true_labels = [self.id2label[l] for l in word_labels]
                    
                    # Extract spans
                    true_spans = self.decode_bioes(true_labels)
                    pred_spans = self.decode_bioes(pred_labels)
                    
                    all_true_spans.extend(true_spans)
                    all_pred_spans.extend(pred_spans)
                    
                    # Reconstruct original message
                    message = self.val_loader.dataset.tokenizer.convert_tokens_to_string(sample_tokens)
                    
                    # Add to analysis records
                    for true_start, true_end, true_type in true_spans:
                        # Get the sentence span
                        span_tokens = sample_tokens[true_start:true_end+1]
                        sentence = self.val_loader.dataset.tokenizer.convert_tokens_to_string(span_tokens)
                        
                        # Check if span was correctly predicted
                        span_found = False
                        pred_type = None
                        for pred_start, pred_end, p_type in pred_spans:
                            if (pred_start, pred_end) == (true_start, true_end):
                                span_found = True
                                pred_type = p_type
                                break
                        
                        analysis_records.append({
                            'message_id': message_id,
                            'message': message,
                            'sentence': sentence,
                            'boundary': f"{true_start}-{true_end}",
                            'true_label': true_type,
                            'predicted_label': pred_type if span_found else 'MISSED',
                            'is_correct_boundary': span_found,
                            'is_correct_type': span_found and true_type == pred_type
                        })
                    
                    # Add false positives to analysis
                    for pred_start, pred_end, pred_type in pred_spans:
                        if not any((true_start == pred_start and true_end == pred_end) 
                                 for true_start, true_end, _ in true_spans):
                            span_tokens = sample_tokens[pred_start:pred_end+1]
                            sentence = self.val_loader.dataset.tokenizer.convert_tokens_to_string(span_tokens)
                            
                            analysis_records.append({
                                'message_id': message_id,
                                'message': message,
                                'sentence': sentence,
                                'boundary': f"{pred_start}-{pred_end}",
                                'true_label': 'NONE',
                                'predicted_label': pred_type,
                                'is_correct_boundary': False,
                                'is_correct_type': False
                            })
                    
                    message_id += 1
        
        # Calculate metrics
        metrics = {
            "boundary": self.calculate_boundary_metrics(all_true_spans, all_pred_spans),
            "type": self.calculate_type_metrics(all_true_spans, all_pred_spans)
        }
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame(analysis_records)
        
        return metrics, analysis_df
    
    def calculate_boundary_metrics(
        self,
        true_spans: List[Tuple[int, int, str]],
        pred_spans: List[Tuple[int, int, str]]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for boundary detection."""
        true_boundaries = {(s, e) for s, e, _ in true_spans}
        pred_boundaries = {(s, e) for s, e, _ in pred_spans}
        
        correct = len(true_boundaries & pred_boundaries)
        precision = correct / len(pred_boundaries) if pred_boundaries else 0
        recall = correct / len(true_boundaries) if true_boundaries else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def calculate_type_metrics(
        self,
        true_spans: List[Tuple[int, int, str]],
        pred_spans: List[Tuple[int, int, str]]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for type classification."""
        true_boundaries = {(s, e) for s, e, _ in true_spans}
        pred_boundaries = {(s, e) for s, e, _ in pred_spans}
        common_boundaries = true_boundaries & pred_boundaries
        
        true_types = []
        pred_types = []
        
        for boundary in common_boundaries:
            true_type = next(t for s, e, t in true_spans if (s, e) == boundary)
            pred_type = next(t for s, e, t in pred_spans if (s, e) == boundary)
            
            # Convert to binary classification (Event vs NonEvent)
            true_types.append(1 if true_type == "Event" else 0)
            pred_types.append(1 if pred_type == "Event" else 0)
        
        if not true_types:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        true_positives = sum(1 for t, p in zip(true_types, pred_types) if t == 1 and p == 1)
        false_positives = sum(1 for t, p in zip(true_types, pred_types) if t == 0 and p == 1)
        false_negatives = sum(1 for t, p in zip(true_types, pred_types) if t == 1 and p == 0)
        
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    

def main():
        
    # Initialize datasets
    train_path = os.path.join("dataset", "train_conll_data.txt")
    test_path = os.path.join("dataset", "test_conll_data.txt")
    train_dataset = NERDataset(train_path)
    val_dataset = NERDataset(test_path)

    # Initialize trainer
    trainer = NERTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name="bert-base-cased",
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3
    )
    # Train and evaluate
    trainer.train()


if __name__ == "__main__":
    main()