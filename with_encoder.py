import os
import math
import json
import random
import argparse
import logging

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast, RobertaModel
from typing import List, Tuple
import numpy as np
from TorchCRF import CRF
from typing import List, Dict, Set, Tuple
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from dataclasses import dataclass


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TAGS = ['O', 
        'B-Event', 'I-Event', 'E-Event', 'S-Event',
        'B-NonEvent', 'I-NonEvent', 'E-NonEvent', 'S-NonEvent']
TAG2IDX = {tag: idx for idx, tag in enumerate(TAGS)}
IDX2TAG = {idx: tag for tag, idx in TAG2IDX.items()}


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--encoder", default='lstm', type=str,
                        help="Encoder to use on top of the pre-trained Embedding. Default: `LSTM`")
    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--train_epochs", default=10, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--scenario',  default='llm-rnn', type=str)
    parser.add_argument('--output_dir',  default='experiments', type=str)
    parser.add_argument('--use_crf', action='store_true',
                        help="Wheter to use CRF as the decode.")
    parser.add_argument('--bidirectional', action='store_true',
                        help="Whether to use bidirectional RNN.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--save_model', action='store_true',
                        help="Whether to save the best checkpoint.")

    args = parser.parse_args()
    model_name = args.model_name_or_path.split('/')[-1]
    encoder_name = 'Bi' + str.upper(args.encoder) if args.bidirectional else str.upper(args.encoder)
    use_crf = "CRF" if args.use_crf else "Linear"
    output_folder = os.path.join(args.output_dir, args.scenario, f'{model_name}_{encoder_name}_{args.num_layers}_{use_crf}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"current scenario - {output_folder}")
    if os.path.exists(os.path.join(output_folder, 'evaluation_score.json')):
        raise ValueError('This scenario has been executed.')
    args.output_dir = output_folder

    return args


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


@dataclass
class EntitySpan:
    start_idx: int
    end_idx: int
    entity_type: str


def extract_valid_spans(labels: List[str]) -> List[EntitySpan]:
    """Extract valid entity spans following BIOES scheme."""
    spans = []
    current_span = None
    current_type = None
    for i, label in enumerate(labels):
        if label == 'O':
            if current_span is not None:
                # Invalid span (no E- tag), discard it
                current_span = None
            continue
            
        prefix = label[0]  # B, I, E, S
        entity_type = label.split('-')[1]  # Event or NonEvent
        
        if prefix in ['B', 'S']:
            if current_span is not None:
                # Invalid span (no E- tag), discard it
                current_span = None
            if prefix == 'S':
                spans.append(EntitySpan(i, i, entity_type))
            else:  # B-
                current_span = i
                current_type = entity_type
        elif prefix == 'I':
            if current_span is None or entity_type != current_type:
                current_span = None  # Invalid I- without proper B- or type mismatch
        elif prefix == 'E':
            if current_span is not None and entity_type == current_type:
                spans.append(EntitySpan(current_span, i, entity_type))
            current_span = None
    
    return spans


def evaluate_predictions(true_labels: List[List[str]], 
                        pred_labels: List[List[str]]) -> Dict:
    """Evaluate both boundary detection and sentence type classification."""
    
    all_true_spans = []
    all_pred_spans = []
    
    # Extract spans for all sentences
    for true_labels, pred_sentence_labels in zip(true_labels, pred_labels):
        true_spans = extract_valid_spans(true_labels)
        pred_spans = extract_valid_spans(pred_sentence_labels)
        
        all_true_spans.append(true_spans)
        all_pred_spans.append(pred_spans)
    
    # Evaluate boundaries
    total_correct = 0
    total_predicted = 0
    total_true = 0
    
    # For classification evaluation
    true_types = []
    pred_types = []
    
    for true_spans, pred_spans in zip(all_true_spans, all_pred_spans):
        # Convert spans to boundary tuples
        true_boundaries = {(span.start_idx, span.end_idx) for span in true_spans}
        pred_boundaries = {(span.start_idx, span.end_idx) for span in pred_spans}
        
        # Get correctly identified boundaries
        correct_boundaries = true_boundaries.intersection(pred_boundaries)
        
        total_correct += len(correct_boundaries)
        total_predicted += len(pred_boundaries)
        total_true += len(true_boundaries)
        
        # Evaluate classification for correct boundaries
        for span_indices in correct_boundaries:
            true_span = next(s for s in true_spans if (s.start_idx, s.end_idx) == span_indices)
            pred_span = next(s for s in pred_spans if (s.start_idx, s.end_idx) == span_indices)
            
            # Event is positive class (1), NonEvent is negative class (0)
            true_types.append(1 if true_span.entity_type == "Event" else 0)
            pred_types.append(1 if pred_span.entity_type == "Event" else 0)
    
    # Calculate boundary metrics
    FP = total_predicted - total_correct
    FN = total_true - total_correct
    boundary_metrics = {
        'precision': total_correct / total_predicted if total_predicted > 0 else 0,
        'recall': total_correct / total_true if total_true > 0 else 0,
        'num_correct': total_correct,
        'num_predicted': total_predicted,
        'num_true': total_true,
        'FP': FP,
        'FN': FN
    }
    
    # Calculate F1 for boundary detection
    if boundary_metrics['precision'] + boundary_metrics['recall'] > 0:
        boundary_metrics['f1'] = (2 * boundary_metrics['precision'] * boundary_metrics['recall'] /
                                (boundary_metrics['precision'] + boundary_metrics['recall']))
    else:
        boundary_metrics['f1'] = 0
    
    # Calculate classification metrics only for correctly identified boundaries
    if true_types:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_types, pred_types, average='binary', zero_division=0,
            labels=[1]  # Ensure we're calculating metrics for Event class
        )
        cm = confusion_matrix(true_types, pred_types)
        TN, FP, FN, TP = cm.ravel()
        accuracy = accuracy_score(true_types, pred_types)
        spesificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        fp_rate = 1 - spesificity
        fn_rate = 1 - recall
        g_mean = math.sqrt(recall * spesificity)
        f1_abs = f1 * boundary_metrics['f1']
    else:
        precision = recall = f1 = accuracy = 0
        TN, FP, FN, TP = 0
        spesificity = fp_rate = fn_rate = 0
        g_mean = f1_abs = 0
    
    classification_metrics = {
        'TP': float(TP),
        'TN': float(TN),
        'FP': float(FP),
        'FN': float(FN),
        'spesificity': spesificity,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'g_mean': g_mean,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_abs': f1_abs,
        'accuracy': accuracy,
        'support': len(true_types)  # Number of correctly identified boundaries
    }

    if boundary_metrics['f1'] + f1_abs > 0:
        f1_f1 = (2 * boundary_metrics['f1'] * f1_abs) / (boundary_metrics['f1'] + f1_abs)
    else:
        f1_f1 = 0
    
    return {
        'boundary': boundary_metrics,
        'classification': classification_metrics,
        'f1_f1': f1_f1
    }


# Example usage:
def evaluation_metrics(true_labels, pred_labels):
    # Evaluate
    metrics = evaluate_predictions(true_labels, pred_labels)
    
    print("\nBoundary Detection Metrics:")
    print(f"Precision: {metrics['boundary']['precision']:.4f}")
    print(f"Recall: {metrics['boundary']['recall']:.4f}")
    print(f"F1: {metrics['boundary']['f1']:.4f}")
    print(f"Correct Boundaries: {metrics['boundary']['num_correct']}")
    print(f"Predicted Boundaries: {metrics['boundary']['num_predicted']}")
    print(f"True Boundaries: {metrics['boundary']['num_true']}")
    
    print("\nClassification Metrics (for correct boundaries):")
    print(f"Precision: {metrics['classification']['precision']:.4f}")
    print(f"Recall: {metrics['classification']['recall']:.4f}")
    print(f"F1: {metrics['classification']['f1']:.4f}")
    print(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
    print(f"Support (correct boundaries): {metrics['classification']['support']}")
    
    return metrics


def log_predictions_to_excel(true_sentences: List[Tuple[List[str], List[str]]], 
                           pred_labels: List[List[str]], 
                           output_dir: str = "error_analysis"):
    """
    Log predictions to Excel files for error analysis.
    
    Args:
        true_sentences: List of tuples (words, labels) from validation set
        predictions: Predictions from SimpleTransformers model
        output_dir: Directory to save the Excel files
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    correct_boundaries_data = []
    incorrect_boundaries_data = []
    
    for msg_id, ((words, true_labels), pred_sentence_labels) in enumerate(zip(true_sentences, pred_labels)):
        # Get the full message
        message = ' '.join(words)
        
        # Extract spans
        true_spans = extract_valid_spans(true_labels)
        pred_spans = extract_valid_spans(pred_sentence_labels)
        
        # Get boundaries
        true_boundaries = {(span.start_idx, span.end_idx) for span in true_spans}
        pred_boundaries = {(span.start_idx, span.end_idx) for span in pred_spans}
        
        # Correctly identified boundaries
        correct_boundaries = true_boundaries.intersection(pred_boundaries)
        
        # Log correct predictions
        for span_indices in correct_boundaries:
            true_span = next(s for s in true_spans if (s.start_idx, s.end_idx) == span_indices)
            pred_span = next(s for s in pred_spans if (s.start_idx, s.end_idx) == span_indices)
            
            # Get the sentence text for this boundary
            sentence_words = words[span_indices[0]:span_indices[1] + 1]
            sentence_text = ' '.join(sentence_words)
            
            correct_boundaries_data.append({
                'message_id': msg_id,
                'message': message,
                'sentence': sentence_text,
                'boundary': f"{span_indices[0]}-{span_indices[1]}",
                'true_label': true_span.entity_type,
                'pred_label': pred_span.entity_type
            })
        
        # Log incorrect predictions
        # Missing predictions (in true but not in pred)
        for span in true_spans:
            if (span.start_idx, span.end_idx) not in pred_boundaries:
                sentence_words = words[span.start_idx:span.end_idx + 1]
                sentence_text = ' '.join(sentence_words)
                
                incorrect_boundaries_data.append({
                    'message_id': msg_id,
                    'message': message,
                    'sentence': sentence_text,
                    'boundary': f"{span.start_idx}-{span.end_idx}",
                    'error_type': 'Missing'
                })
        
        # False predictions (in pred but not in true)
        for span in pred_spans:
            if (span.start_idx, span.end_idx) not in true_boundaries:
                sentence_words = words[span.start_idx:span.end_idx + 1]
                sentence_text = ' '.join(sentence_words)
                
                incorrect_boundaries_data.append({
                    'message_id': msg_id,
                    'message': message,
                    'sentence': sentence_text,
                    'boundary': f"{span.start_idx}-{span.end_idx}",
                    'error_type': 'False'
                })
    
    # Create DataFrames and save to Excel
    if correct_boundaries_data:
        correct_df = pd.DataFrame(correct_boundaries_data)
        correct_df.to_excel(
            os.path.join(output_dir, 'correct_predictions.xlsx'),
            index=False
        )
    
    if incorrect_boundaries_data:
        incorrect_df = pd.DataFrame(incorrect_boundaries_data)
        incorrect_df.to_excel(
            os.path.join(output_dir, 'incorrect_predictions.xlsx'),
            index=False
        )
    
    print(f"Logged {len(correct_boundaries_data)} correct predictions")
    print(f"Logged {len(incorrect_boundaries_data)} incorrect predictions")
    print(f"Files saved in directory: {output_dir}")


def log_word_level_predictions(true_sentences: List[Tuple[List[str], List[str]]], 
                             pred_labels: List[List[str]], 
                             output_dir: str = "error_analysis"):
    """
    Log word-level predictions for detailed error analysis.
    
    Args:
        true_sentences: List of tuples (words, labels) from validation set
        predictions: Predictions from SimpleTransformers model
        output_dir: Directory to save the Excel files
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    word_level_data = []
    
    for msg_id, ((words, true_labels), pred_sentence_labels) in enumerate(zip(true_sentences, pred_labels)):
        # Get the full message
        message = ' '.join(words)
        
        # Extract spans for identifying whether a token is part of a valid boundary
        true_spans = extract_valid_spans(true_labels)
        pred_spans = extract_valid_spans(pred_sentence_labels)
        
        # Get valid boundaries
        true_boundaries = {(span.start_idx, span.end_idx): span.entity_type 
                         for span in true_spans}
        pred_boundaries = {(span.start_idx, span.end_idx): span.entity_type 
                         for span in pred_spans}
        
        # For each word in the message
        for idx, (word, true_label, pred_label) in enumerate(zip(words, true_labels, pred_sentence_labels)):
            # Check if this token is part of valid boundaries
            true_entity = None
            pred_entity = None
            
            # Find if token is part of a true boundary
            for (start, end), entity_type in true_boundaries.items():
                if start <= idx <= end:
                    true_entity = entity_type
                    break
            
            # Find if token is part of a predicted boundary
            for (start, end), entity_type in pred_boundaries.items():
                if start <= idx <= end:
                    pred_entity = entity_type
                    break
            
            is_boundary_correct = None
            if true_entity is not None:
                # Token should be part of a boundary
                if pred_entity is not None:
                    # Boundary detected, check if entity type matches
                    is_boundary_correct = (true_entity == pred_entity)
                else:
                    # Boundary missed
                    is_boundary_correct = False
            else:
                # Token should not be part of a boundary
                is_boundary_correct = (pred_entity is None)
            
            word_level_data.append({
                'message_id': msg_id,
                'message': message,
                'token': word,
                'token_index': idx,
                'true_label': true_label,
                'pred_label': pred_label,
                'is_in_true_boundary': true_entity is not None,
                'true_entity_type': true_entity if true_entity else 'None',
                'is_in_pred_boundary': pred_entity is not None,
                'pred_entity_type': pred_entity if pred_entity else 'None',
                'is_boundary_correct': is_boundary_correct,
                'error_type': get_error_type(true_label, pred_label)
            })
    
    # Create DataFrame and save to Excel
    word_level_df = pd.DataFrame(word_level_data)
    
    # Add color formatting for easier visualization
    def color_incorrect_predictions(row):
        if not row['is_boundary_correct']:
            return ['background-color: #FFB6B6'] * len(row)
        return [''] * len(row)
    
    # Apply conditional formatting
    styled_df = word_level_df.style.apply(color_incorrect_predictions, axis=1)
    
    # Save to Excel with formatting
    with pd.ExcelWriter(os.path.join(output_dir, 'word_level_predictions.xlsx'),
                       engine='openpyxl') as writer:
        styled_df.to_excel(writer, index=False)
    
    # Create summary of error types
    error_summary = pd.DataFrame(word_level_df[~word_level_df['is_boundary_correct']]['error_type'].value_counts())
    error_summary.columns = ['count']
    error_summary.to_excel(os.path.join(output_dir, 'word_level_error_type_summary.xlsx'))
    
    print(f"Logged {len(word_level_data)} word-level predictions")
    print(f"Files saved in directory: {output_dir}")


def get_error_type(true_label: str, pred_label: str) -> str:
    """
    Determine the type of error for a token prediction.
    """
    if true_label == pred_label:
        return 'Correct'
    
    def get_prefix_type(label):
        if label == 'O':
            return 'O'
        prefix = label[0]
        entity = label.split('-')[1]
        return f"{prefix}-{entity}"
    
    true_type = get_prefix_type(true_label)
    pred_type = get_prefix_type(pred_label)
    
    if true_label == 'O':
        return f"False_{pred_type}"
    elif pred_label == 'O':
        return f"Missed_{true_type}"
    else:
        # Both are tags but different
        true_prefix, true_entity = true_label.split('-')
        pred_prefix, pred_entity = pred_label.split('-')
        
        if true_entity != pred_entity:
            return f"Wrong_Entity_{true_entity}_as_{pred_entity}"
        else:
            return f"Wrong_Tag_{true_prefix}_as_{pred_prefix}"


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
            return_tensors='pt',
        )
        
        # Convert tags to ids and align with wordpieces
        word_ids = tokenized.word_ids()
        label_ids = []
        # Convert word_ids to a list of integers, using -1 for None
        word_ids_int = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # special tokens
                word_ids_int.append(-1)  # Use -1 to represent None
            elif word_id != previous_word_id:
                label_ids.append(TAG2IDX[tags[word_id]])
                word_ids_int.append(word_id)
            else:
                label_ids.append(-100)
                word_ids_int.append(word_id)
           
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids),
            'word_ids': torch.tensor(word_ids_int),
        }

class SequenceLabelingModel(nn.Module):
    def __init__(self,
                 args,
                 num_tags: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=384 if args.bidirectional else 768,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            batch_first=True,
            dropout=dropout if args.num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, num_tags)
        # Decoder
        self.use_crf = args.use_crf
        if args.use_crf:
            self.crf = CRF(num_tags)
        else:
            self.softmax = nn.LogSoftmax(dim=2)
            
        
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
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
                return loss
            else:  # Inference
                mask = attention_mask.bool()
                predictions = self.crf.decode(emissions, mask=mask)
                return predictions
        else:
            if labels is not None:  # Training
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, emissions.shape[-1])
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
                return loss
            else:  # Inference
                logits = self.softmax(emissions)
                return torch.argmax(logits, dim=-1)


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


def evaluate_model(model, test_loader, test_dataset: DroneLogDataset, device):
    """Evaluate model on test set"""
    test_dataset = test_dataset.data
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
                word_preds = []
                for i, word_ids in enumerate(batch["word_ids"]):
                    word_pred = []
                    prev_word_id = None
                    for j, word_id in enumerate(word_ids):
                        if word_id == -1 or word_id == prev_word_id:
                            continue  # Ignore padding/subwords
                        word_pred.append(predictions[i, j].item())
                        prev_word_id = word_id
                    word_preds.append(word_pred)
                pred_tags = [[IDX2TAG[p] for p in pred] for pred in word_preds]
            
            all_predictions.extend(pred_tags)
    all_labels = [labels for _, labels in test_dataset]
    metrics = evaluation_metrics(all_labels, all_predictions)

    return all_predictions, all_labels, metrics

def main():
    # initialization
    args = init_args()
    seed_everything(args.seed)
    # Hyperparameters
    pretrained_model = args.model_name_or_path
    max_length = 128
    batch_size = args.train_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.train_epochs
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True)
    
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
        args=args,
        num_tags=len(TAGS),
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, num_epochs, device)
    
    # Load best model and evaluate
    # model.load_state_dict(torch.load('best_model.pt'))
    pred_labels, true_labels, metrics = evaluate_model(model, test_loader, test_dataset, device)
    arguments_dict = vars(args)
    metrics["scenario_args"] = arguments_dict
    logger.info(f'Eval Score: \n{metrics}')
    with open(os.path.join(args.output_dir, "evaluation_score.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    with open(os.path.join(args.output_dir, "true_labels.json"), "w") as f:
            json.dump(true_labels, f, indent=4)
    with open(os.path.join(args.output_dir, "pred_labels.json"), "w") as f:
            json.dump(pred_labels, f, indent=4)

    # Log predictions for error analysis
    log_predictions_to_excel(
        true_sentences=test_dataset.data,
        pred_labels=pred_labels,
        output_dir=args.output_dir
    )

    # Log word-level predictions for detailed error analysis
    log_word_level_predictions(
        true_sentences=test_dataset.data,
        pred_labels=pred_labels,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()