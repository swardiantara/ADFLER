from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast
import torch
import os
from typing import List, Dict, Tuple
# from seqeval.metrics import classification_report


class NERDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizerFast,
        max_length: int = 128,
        label_to_id: Dict[str, int] = None
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Read CoNLL data
        self.sentences, self.labels = self._read_conll_file(data_path) if data_path is not None else None
        
        # Create label vocabulary if not provided
        if label_to_id is None:
            unique_labels = set()
            for label_seq in self.labels:
                unique_labels.update(label_seq)
            self.label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label_to_id = label_to_id
            
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
    def _read_conll_file(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Read CoNLL format file and return sentences and labels."""
        sentences, labels = [], []
        current_sentence, current_labels = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split line into tokens (assuming space/tab-separated format)
                    parts = line.split()
                    if len(parts) >= 2:  # Ensure we have both token and label
                        token, label = parts[0], parts[-1]  # Take last column as label
                        current_sentence.append(token)
                        current_labels.append(label)
                elif current_sentence:  # Empty line indicates sentence boundary
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
                    
        # Add last sentence if file doesn't end with empty line
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
            
        return sentences, labels
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Process a single item with wordpiece tokenization and label alignment."""
        words = self.sentences[idx]
        labels = self.labels[idx]
        
        # Initialize lists for aligned tokens and labels
        wordpiece_tokens = ["[CLS]"]
        aligned_labels = ["O"]
        word_ids = [-1]  # For tracking original word positions
        
        # Process each word and its label
        for word_idx, (word, label) in enumerate(zip(words, labels)):
            word_pieces = self.tokenizer.tokenize(word)
            
            if len(word_pieces) == 0:
                continue
                
            # Handle label alignment based on number of wordpieces
            if len(word_pieces) == 1:
                wordpiece_tokens.extend(word_pieces)
                aligned_labels.append(label)
                word_ids.append(word_idx)
            else:
                wordpiece_tokens.extend(word_pieces)
                
                # Adjust BIOES tags for split words
                if label.startswith('S-'):
                    entity_type = label[2:]
                    aligned_labels.append(f'B-{entity_type}')
                    for i in range(1, len(word_pieces) - 1):
                        aligned_labels.append(f'I-{entity_type}')
                    aligned_labels.append(f'E-{entity_type}')
                elif label.startswith('B-'):
                    entity_type = label[2:]
                    aligned_labels.append(f'B-{entity_type}')
                    for _ in range(len(word_pieces) - 1):
                        aligned_labels.append(f'I-{entity_type}')
                elif label.startswith('E-'):
                    entity_type = label[2:]
                    for _ in range(len(word_pieces) - 1):
                        aligned_labels.append(f'I-{entity_type}')
                    aligned_labels.append(f'E-{entity_type}')
                elif label.startswith('I-'):
                    entity_type = label[2:]
                    for _ in range(len(word_pieces)):
                        aligned_labels.append(f'I-{entity_type}')
                else:  # 'O' label
                    for _ in range(len(word_pieces)):
                        aligned_labels.append('O')
                
                # Track original word positions for all pieces
                word_ids.extend([word_idx] * len(word_pieces))
        
        # Add [SEP] token
        wordpiece_tokens.append("[SEP]")
        aligned_labels.append("O")
        word_ids.append(-1)
        
        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(wordpiece_tokens)
        label_ids = [self.label_to_id[label] for label in aligned_labels]
        
        # Create attention mask and pad sequences
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            label_ids = label_ids + [self.label_to_id["O"]] * padding_length
            word_ids = word_ids + [-1] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_ids = label_ids[:self.max_length]
            word_ids = word_ids[:self.max_length]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(label_ids),
            "word_ids": torch.tensor(word_ids),
            "original_words": self.sentences[idx],  # Keep original words for evaluation
            "original_labels": self.labels[idx]     # Keep original labels for evaluation
        }
    


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

