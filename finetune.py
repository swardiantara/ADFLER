import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from typing import List, Tuple, Dict, Set
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label2id = {
            'O': 0, 
            'B-Event': 1, 'I-Event': 2, 'E-Event': 3, 'S-Event': 4,
            'B-NonEvent': 5, 'I-NonEvent': 6, 'E-NonEvent': 7, 'S-NonEvent': 8
        }
id2label = {v: k for k, v in self.label2id.items()}
@dataclass
class EntitySpan:
    start_idx: int
    end_idx: int
    entity_type: str

class CoNLLDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.sentences = []  # List of word-level sentences
        self.labels = []     # List of word-level labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Define label scheme
        self.label2id = label2id
        self.id2label = id2label
        
        self._read_conll_file(file_path)
    
    def _read_conll_file(self, file_path: str):
        current_sentence = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split()
                    current_sentence.append(token)
                    current_labels.append(self.label2id[label])
                elif current_sentence:
                    self.sentences.append(current_sentence)
                    self.labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            
            if current_sentence:
                self.sentences.append(current_sentence)
                self.labels.append(current_labels)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        word_labels = self.labels[idx]
        
        # Tokenize and align labels
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with wordpiece tokens
        word_ids = tokenized.word_ids()
        aligned_labels = self._align_labels_with_tokens(word_ids, word_labels)
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels),
            'word_ids': word_ids,  # Keep word_ids for reconstruction
            'original_words': words,
            'original_labels': word_labels
        }
    
    def _align_labels_with_tokens(self, word_ids, word_labels):
        aligned_labels = [-100] * len(word_ids)  # -100 is ignored in loss calculation
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:
                # Special handling for first token of word
                if i == 0 or word_ids[i - 1] != word_idx:
                    aligned_labels[i] = word_labels[word_idx]
                else:
                    # For other tokens of same word, maintain the label type but convert to I- tag
                    prev_label = word_labels[word_idx]
                    if prev_label in [1, 4]:  # B- or S- Event
                        aligned_labels[i] = 2  # I-Event
                    elif prev_label in [5, 8]:  # B- or S- NonEvent
                        aligned_labels[i] = 6  # I-NonEvent
                    else:
                        aligned_labels[i] = prev_label
        
        return aligned_labels

class SequenceLabelingTrainer:
    def __init__(self, model_name: str, num_labels: int, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).to(device)
        self.device = device
    
    def train(self, train_dataloader, val_dataloader, num_epochs, learning_rate):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Training Loss: {total_loss/len(train_dataloader):.4f}")
            logger.info(f"Validation Metrics: {val_metrics}")
    
    def evaluate(self, dataloader) -> Dict:
        self.model.eval()
        all_word_predictions = []
        all_word_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                
                # Get token-level predictions
                token_predictions = torch.argmax(outputs.logits, dim=2)
                
                # Reconstruct word-level predictions for each sentence in batch
                for pred, word_ids, orig_labels in zip(
                    token_predictions, batch['word_ids'], batch['original_labels']
                ):
                    word_preds = self._reconstruct_word_predictions(
                        pred.cpu().numpy(), word_ids, len(orig_labels)
                    )
                    all_word_predictions.extend(word_preds)
                    all_word_labels.extend(orig_labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_word_labels, all_word_predictions)
        return metrics
    
    def _reconstruct_word_predictions(self, token_preds, word_ids, num_words) -> List[int]:
        word_preds = [-1] * num_words
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                # Take the prediction of the first token of each word
                if word_preds[word_idx] == -1:
                    word_preds[word_idx] = token_preds[token_idx]
        
        return word_preds
    
    def _extract_valid_spans(self, labels: List[int]) -> List[EntitySpan]:
        spans = []
        current_span = None
        current_type = None
        
        def is_same_type(label1, label2):
            # Check if two labels belong to the same entity type (Event/NonEvent)
            return (label1 in [1,2,3,4] and label2 in [1,2,3,4]) or \
                   (label1 in [5,6,7,8] and label2 in [5,6,7,8])
        
        for i, label in enumerate(labels):
            if label in [1, 4]:  # B- or S-
                if current_span:
                    # Invalid span (no E- tag), discard it
                    current_span = None
                if label == 4:  # S-
                    entity_type = "Event" if label == 4 else "NonEvent"
                    spans.append(EntitySpan(i, i, entity_type))
                else:  # B-
                    current_span = i
                    current_type = "Event" if label in [1,2,3,4] else "NonEvent"
            elif label in [2]:  # I-
                if current_span is not None and is_same_type(labels[i-1], label):
                    continue
                else:
                    current_span = None  # Invalid I- without proper B-
            elif label in [3]:  # E-
                if current_span is not None and is_same_type(labels[i-1], label):
                    entity_type = "Event" if label in [1,2,3,4] else "NonEvent"
                    spans.append(EntitySpan(current_span, i, entity_type))
                current_span = None
            elif label == 0:  # O
                current_span = None
        
        return spans
    
    def _calculate_metrics(self, true_labels: List[int], pred_labels: List[int]) -> Dict:
        # Extract valid spans following BIOES scheme
        true_spans = self._extract_valid_spans(true_labels)
        pred_spans = self._extract_valid_spans(pred_labels)
        
        # Evaluate boundary detection
        true_boundaries = {(span.start_idx, span.end_idx) for span in true_spans}
        pred_boundaries = {(span.start_idx, span.end_idx) for span in pred_spans}
        
        correct_boundaries = true_boundaries.intersection(pred_boundaries)
        
        boundary_precision = len(correct_boundaries) / len(pred_boundaries) if pred_boundaries else 0
        boundary_recall = len(correct_boundaries) / len(true_boundaries) if true_boundaries else 0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) \
                     if boundary_precision + boundary_recall > 0 else 0
        
        # Evaluate classification only for correctly identified boundaries
        true_types = []
        pred_types = []
        
        for span_indices in correct_boundaries:
            true_span = next(s for s in true_spans if (s.start_idx, s.end_idx) == span_indices)
            pred_span = next(s for s in pred_spans if (s.start_idx, s.end_idx) == span_indices)
            
            true_types.append(1 if true_span.entity_type == "Event" else 0)
            pred_types.append(1 if pred_span.entity_type == "Event" else 0)
        
        if true_types:  # Only calculate if there are correctly identified boundaries
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_types, pred_types, average='binary', zero_division=0
            )
            accuracy = accuracy_score(true_types, pred_types)
        else:
            precision = recall = f1 = accuracy = 0
        
        return {
            'boundary': {
                'precision': boundary_precision,
                'recall': boundary_recall,
                'f1': boundary_f1,
                'num_correct': len(correct_boundaries),
                'num_predicted': len(pred_boundaries),
                'num_true': len(true_boundaries)
            },
            'classification': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'support': len(true_types)  # Number of correctly identified boundaries
            }
        }

def main():
    # Initialize trainer with any pre-trained model
    model_name = 'bert-base-cased'  # Can be changed to any HuggingFace model
    num_labels = 9  # O + 4 BIOES tags * 2 classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = SequenceLabelingTrainer(model_name, num_labels, device)
    
    # Create datasets
    train_path = os.path.join("dataset", "train_conll_data.txt")
    test_path = os.path.join("dataset", "test_conll_data.txt")
    train_dataset = CoNLLDataset(train_path, trainer.tokenizer)
    val_dataset = CoNLLDataset(test_path, trainer.tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Train and evaluate
    trainer.train(train_dataloader, val_dataloader, num_epochs=5, learning_rate=2e-5)

if __name__ == '__main__':
    main()