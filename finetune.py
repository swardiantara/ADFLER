import os
import pandas as pd
import numpy as np
from simpletransformers.ner import NERModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

@dataclass
class EntitySpan:
    start_idx: int
    end_idx: int
    entity_type: str

def read_conll_file(file_path: str) -> List[Tuple[List[str], List[str]]]:
    sentences = []
    current_words = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word, label = line.split()
                current_words.append(word)
                current_labels.append(label)
            elif current_words:
                sentences.append((current_words, current_labels))
                current_words = []
                current_labels = []
    
    if current_words:
        sentences.append((current_words, current_labels))
    
    return sentences

def convert_to_simpletransformers_format(sentences: List[Tuple[List[str], List[str]]]) -> List[Dict]:
    formatted_data = []
    
    for words, labels in sentences:
        sentence_data = []
        for i, (word, label) in enumerate(zip(words, labels)):
            sentence_data.append({
                'words': word,
                'labels': label,
                'sentence_id': len(formatted_data)
            })
        formatted_data.extend(sentence_data)
    
    return formatted_data

def extract_valid_spans(words: List[str], labels: List[str]) -> List[EntitySpan]:
    spans = []
    current_span = None
    current_type = None
    
    def get_entity_type(label: str) -> str:
        return label.split('-')[1] if '-' in label else 'O'
    
    for i, label in enumerate(labels):
        prefix = label[0] if '-' in label else 'O'
        entity_type = get_entity_type(label)
        
        if prefix in ['B', 'S']:
            if current_span:
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
        elif prefix == 'O':
            current_span = None
    
    return spans

def evaluate_predictions(true_sentences: List[Tuple[List[str], List[str]]], 
                        pred_labels: List[List[str]]) -> Dict:
    all_true_spans = []
    all_pred_spans = []
    
    # Extract spans for all sentences
    for (words, true_labels), pred_sentence_labels in zip(true_sentences, pred_labels):
        true_spans = extract_valid_spans(words, true_labels)
        pred_spans = extract_valid_spans(words, pred_sentence_labels)
        
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
        true_boundaries = {(span.start_idx, span.end_idx) for span in true_spans}
        pred_boundaries = {(span.start_idx, span.end_idx) for span in pred_spans}
        
        correct_boundaries = true_boundaries.intersection(pred_boundaries)
        
        total_correct += len(correct_boundaries)
        total_predicted += len(pred_boundaries)
        total_true += len(true_boundaries)
        
        # Evaluate classification for correct boundaries
        for span_indices in correct_boundaries:
            true_span = next(s for s in true_spans if (s.start_idx, s.end_idx) == span_indices)
            pred_span = next(s for s in pred_spans if (s.start_idx, s.end_idx) == span_indices)
            
            true_types.append(1 if true_span.entity_type == "Event" else 0)
            pred_types.append(1 if pred_span.entity_type == "Event" else 0)
    
    # Calculate boundary metrics
    boundary_precision = total_correct / total_predicted if total_predicted > 0 else 0
    boundary_recall = total_correct / total_true if total_true > 0 else 0
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) \
                 if boundary_precision + boundary_recall > 0 else 0
    
    # Calculate classification metrics
    if true_types:
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
            'num_correct': total_correct,
            'num_predicted': total_predicted,
            'num_true': total_true
        },
        'classification': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': len(true_types)
        }
    }

def main():
    # Read data
    train_path = os.path.join("dataset", "train_conll_data.txt")
    test_path = os.path.join("dataset", "test_conll_data.txt")
    train_sentences = read_conll_file(train_path)
    val_sentences = read_conll_file(test_path)
    
    # Convert to SimpleTransformers format
    train_data = convert_to_simpletransformers_format(train_sentences)
    val_data = convert_to_simpletransformers_format(val_sentences)
    
    # Get unique labels
    labels = sorted(list({item['labels'] for item in train_data}))
    
    # Define model arguments
    model_args = {
        'num_train_epochs': 5,
        'learning_rate': 2e-5,
        'overwrite_output_dir': True,
        'train_batch_size': 16,
        'eval_batch_size': 16,
        'output_dir': 'simple-trans'
    }
    
    # Initialize model (can use any transformer model)
    model = NERModel(
        'bert',  # Model type (can be roberta, xlnet, etc.)
        'bert-base-uncased',  # Model name
        labels=labels,
        args=model_args
    )
    
    # Train the model
    model.train_model(train_data)
    
    # Make predictions on validation set
    predictions, _ = model.predict([' '.join(words) for words, _ in val_sentences])
    
    # Evaluate
    metrics = evaluate_predictions(val_sentences, predictions)
    print("Validation Metrics:", metrics)

if __name__ == '__main__':
    main()