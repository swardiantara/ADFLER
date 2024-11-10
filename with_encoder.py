import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import numpy as np
from TorchCRF import CRF
import logging
import json
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TAGS = ['O', 
        'B-Event', 'I-Event', 'E-Event', 'S-Event',
        'B-NonEvent', 'I-NonEvent', 'E-NonEvent', 'S-NonEvent']
TAG2IDX = {tag: idx for idx, tag in enumerate(TAGS)}
IDX2TAG = {idx: tag for tag, idx in TAG2IDX.items()}

class DroneLogDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_conll_data(data_path)
    
    def load_conll_data(self, data_path: str) -> List[Tuple[List[str], List[str]]]:
        """Load CoNLL format data"""
        data = []
        current_tokens = []
        current_tags = []
        
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    if current_tokens:
                        data.append((current_tokens, current_tags))
                        current_tokens = []
                        current_tags = []
                else:
                    token, tag = line.split()
                    current_tokens.append(token)
                    current_tags.append(tag)
                    
        if current_tokens:  # handle the last sequence
            data.append((current_tokens, current_tags))
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        
        # Tokenize text
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert tags to ids and align with wordpieces
        label_ids = []
        word_ids = tokenized.word_ids()
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # special tokens
            else:
                label_ids.append(TAG2IDX[tags[word_idx]])
                
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }

class SequenceLabelingModel(nn.Module):
    def __init__(self, 
                 pretrained_model: str,
                 hidden_size: int,
                 num_layers: int,
                 num_tags: int,
                 dropout: float = 0.1,
                 use_crf: bool = True):
        super().__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(pretrained_model)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_tags)
            self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
        else:
            self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
            self.softmax = nn.LogSoftmax(dim=2)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None):
        
        # Get BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(sequence_output)
        
        # Decoding
        emissions = self.hidden2tag(lstm_out)
        
        if self.use_crf:
            if labels is not None:  # Training
                mask = attention_mask.bool()
                loss = -self.crf(emissions, labels, mask=mask)
                return loss
            else:  # Inference
                mask = attention_mask.bool()
                predictions = self.crf.decode(emissions, mask=mask)
                return predictions
        else:
            logits = self.softmax(emissions)
            if labels is not None:  # Training
                loss_fct = nn.NLLLoss(ignore_index=-100)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = labels.view(-1)
                loss = loss_fct(active_logits, active_labels)
                return loss
            else:  # Inference
                return torch.argmax(logits, dim=2)

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device):
    """Training loop with validation"""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss = model(input_ids, attention_mask, labels)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Save best model
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), 'best_model.pt')
            
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Average training loss: {avg_train_loss:.4f}')
        logger.info(f'Average validation loss: {avg_val_loss:.4f}')

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            predictions = model(input_ids, attention_mask)
            
            # Convert predictions to tags
            if model.use_crf:
                pred_tags = [[IDX2TAG[p] for p in pred] for pred in predictions]
            else:
                predictions = predictions.cpu().numpy()
                pred_tags = [[IDX2TAG[p] for p in pred] for pred in predictions]
            
            # Convert labels to tags
            true_tags = [[IDX2TAG[l.item()] if l.item() != -100 else 'O' 
                         for l in label] for label in labels]
            
            all_predictions.extend(pred_tags)
            all_labels.extend(true_tags)
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(
        [tag for seq in all_labels for tag in seq],
        [tag for seq in all_predictions for tag in seq]
    ))

def main():
    # Hyperparameters
    pretrained_model = "bert-base-uncased"
    max_length = 128
    batch_size = 8
    hidden_size = 384
    num_layers = 2
    learning_rate = 2e-5
    num_epochs = 5
    use_crf = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # Load datasets
    train_path = os.path.join("dataset", "train_conll_data.txt")
    test_path = os.path.join("dataset", "test_conll_data.txt")
    train_dataset = DroneLogDataset(train_path, tokenizer, max_length)
    val_dataset = DroneLogDataset(test_path, tokenizer, max_length)
    test_dataset = DroneLogDataset(test_path, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SequenceLabelingModel(
        pretrained_model=pretrained_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_tags=len(TAGS),
        use_crf=use_crf
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, num_epochs, device)
    
    # Load best model and evaluate
    # model.load_state_dict(torch.load('best_model.pt'))
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()