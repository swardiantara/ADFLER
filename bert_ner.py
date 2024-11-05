import os
import json

import torch
from itertools import chain
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

from src.eval_utils import log_errors_for_analysis, evaluate_sbd_boundary_only, evaluate_classification_correct_boundaries

class DroneLogDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = self.read_conll_file(data_path) if data_path is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {'O': 0, 'B-Event': 1, 'I-Event': 2, 'E-Event': 3, 'S-Event': 4, 'B-NonEvent': 5, 'I-NonEvent': 6, 'E-NonEvent': 7, 'S-NonEvent': 8}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def read_conll_file(self, file_path):
        """Read CoNLL format file"""
        data = []
        current_words = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    if current_words:
                        data.append((current_words, current_labels))
                        current_words = []
                        current_labels = []
                else:
                    splits = line.split()
                    current_words.append(splits[0])
                    current_labels.append(splits[-1])
                    
        if current_words:
            data.append((current_words, current_labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, labels = self.data[idx]
        
        # Tokenize words and align labels
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert labels to ids and align with tokens
        label_ids = []
        word_ids = encoded.word_ids()
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # special tokens
            else:
                label_ids.append(self.label2id[labels[word_idx]])

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids)
        }

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
        train_dataset = DroneLogDataset(train_path, self.tokenizer)
        val_dataset = DroneLogDataset(val_path, self.tokenizer)
        
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
            val_loss, _, _ = self.evaluate(val_loader)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

    def evaluate(self, data_loader):
        self.model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        label2id = DroneLogDataset(None, self.tokenizer).label2id
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
                preds = torch.argmax(outputs.logits, dim=2)
                
                # Convert predictions and labels to list, filtering out padding (-100)
                for pred, label in zip(preds, labels):
                    valid_indices = label != -100
                    all_preds.append(pred[valid_indices].cpu().numpy())
                    all_labels.append(label[valid_indices].cpu().numpy())

        all_preds = [[id2label[idx] for idx in sample] for sample in all_preds]
        all_labels = [[id2label[idx] for idx in sample] for sample in all_labels]

        return total_val_loss / len(data_loader), all_preds, all_labels
    

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
                id2label = DroneLogDataset(None, self.tokenizer).id2label
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

# Example usage
def main():
    # Initialize the model
    ner_model = DroneLogNER(device='cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join('dataset', 'train_conll_data.txt')
    test_path = os.path.join('dataset', 'test_conll_data.txt')

    # Train the model
    ner_model.train(
        train_path=train_path,
        val_path=test_path,
        epochs=5
    )

    # Evaluation
    val_dataset = DroneLogDataset(test_path, ner_model.tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16)
    _, all_pred_tags, all_true_tags = ner_model.evaluate(val_loader)
    sbd_scores = evaluate_sbd_boundary_only(all_true_tags, all_pred_tags)
    classification_scores = evaluate_classification_correct_boundaries(all_true_tags, all_pred_tags)
    eval_score = {
        "sbd_score": sbd_scores,
        "classification_score": classification_scores
    }
    with open("evaluation_score.json", "w") as f:
        json.dump(eval_score, f, indent=4)

    val_dataset = DroneLogDataset(test_path, ner_model.tokenizer).read_conll_file(test_path)
    logs = log_errors_for_analysis(all_pred_tags, val_dataset)
    with open("error_analysis_logs.json", "w") as f:
        json.dump(logs, f, indent=4)

    # Make predictions
    # Load your trained model if necessary
    # ner_model.model.load_state_dict(torch.load('best_model.pt'))
    sample_text = "Motor speed error. Land or return to home promptly. After powering off the aircraft, replace the propeller on the beeping ESC. If the issue persists, contact DJI Support."
    predictions = ner_model.predict(sample_text)
    print("\nPredictions:")
    for token, label in predictions:
        print(f"{token}: {label}")


if __name__ == "__main__":
    main()