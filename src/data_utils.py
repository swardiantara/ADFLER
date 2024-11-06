import os
from typing import List, Dict, Tuple
# from seqeval.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer

class NERDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        label2id: Dict[str, int] = None
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Read CoNLL data
        self.sentences, self.labels = self._read_conll_file(data_path) if data_path is not None else None
        
        # Create label vocabulary if not provided
        if label2id is None:
            unique_labels = set()
            for label_seq in self.labels:
                unique_labels.update(label_seq)
            self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label2id = label2id
            
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
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
        label_ids = [self.label2id[label] for label in aligned_labels]
        
        # Create attention mask and pad sequences
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            label_ids = label_ids + [self.label2id["O"]] * padding_length
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
    def __init__(self, args, data_path, tokenizer: AutoTokenizer, label2id: Dict[str, int] = None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = args.max_seq_length
        self.align_mode = args.align_label
        # Read the data from the file
        self.data = self._read_conll_file(data_path) if data_path is not None else None
        # Create label vocabulary if not provided
        if label2id is None:
            unique_labels = set()
            for label_seq in self.labels:
                unique_labels.update(label_seq)
            self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _read_conll_file(self, file_path):
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
    
    def align_labels_with_padding(self, words: List[str], labels: List[str]):
        """
        Align labels with wordpiece tokens and handle special tokens
        Returns aligned label ids and a mapping to original word indices
        """
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label_ids = []
        original_word_ids = []
        prev_word_idx = None

        for word_idx in encoded.word_ids():
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
                original_word_ids.append(-1)
            else:
                if word_idx != prev_word_idx:
                    # First token of word gets the actual label
                    label_ids.append(self.label2id[labels[word_idx]])
                else:
                    # Additional wordpieces get the PAD tag
                    label_ids.append(self.label2id['PAD'])
                original_word_ids.append(word_idx)
                prev_word_idx = word_idx

        return encoded, label_ids, original_word_ids

    def __getitem__(self, idx):
        words, labels = self.data[idx]
        
        if self.align_mode == 'padding':
            encoded, label_ids, original_word_ids = self.align_labels_with_padding(words, labels)
        elif self.align_mode == 'bioes':
            encoded, label_ids, original_word_ids = self.align_labels_with_padding(words, labels)
        else:
            NotImplementedError
        # Assert the dimension of tokenized input and aligned label
        assert len(encoded['input_ids']) == len(label_ids), f"The dimension of `encoded['input_ids']` of {len(encoded['input_ids'])} is not the same with `label_ids` of {len(label_ids)}"
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids),
            'word_ids': torch.tensor(original_word_ids),  # Keep track of original word mapping
            # "original_words": words,  # Keep original words for evaluation
            # "original_labels": labels     # Keep original labels for evaluation
        }

    def reconstruct_labels_padding(self, token_labels: List[int], word_ids: List[int]) -> List[str]:
        """
        Reconstruct original word-level labels from token-level predictions
        
        Args:
            token_labels: Predicted labels for each token
            word_ids: Mapping from tokens to original words
            
        Returns:
            List of labels at word level
        """
        reconstructed_labels = []
        current_word_idx = -1
        
        for word_idx, label_id in zip(word_ids, token_labels):
            if (word_idx == -1) or (label_id == self.label2id['PAD']):
                continue
                
            if word_idx != current_word_idx:
                reconstructed_labels.append(self.id2label[label_id])
                current_word_idx = word_idx
                
        return reconstructed_labels
    

def sanity_check(dataset, idx):
    """
    Perform sanity check on a specific example in the dataset
    
    Args:
        dataset: NERDataset instance
        idx: index of example to check
    """
    # Get the item
    item = dataset[idx]
    
    # Get the original tokens
    tokens = dataset.tokenizer.convert_ids_to_tokens(item['input_ids'])
    labels = item['labels'].tolist()
    
    # Create id2label mapping
    id2label = {v: k for k, v in dataset.label2id.items()}
    
    # Print alignment
    print("Token\t\tLabel")
    print("-" * 40)
    for token, label_id in zip(tokens, labels):
        # Convert label_id to label string
        label = id2label[label_id] if label_id != -100 else 'PAD/SUBWORD'
        # Adjust spacing for better visualization
        print(f"{token:<15} {label}")
        
    # Print some statistics
    print("\nStatistics:")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Number of actual labels (excluding -100): {sum(1 for l in labels if l != -100)}")