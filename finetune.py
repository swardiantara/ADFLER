import os
import sys
import random
import json
import argparse
import math

import torch
import pandas as pd
import numpy as np
from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str,
                        help="Path to pre-trained model or shortcut name")
    # other parameters
    parser.add_argument('--train_dataset',  default='original', type=str,
                        help="Whether to use original or augmented dataset. Options: [`original`, `aug-20`, `aug-40`, `rem-100`]. Default: `original`")
    parser.add_argument('--eval_dataset',  default='original', type=str,
                        help="Whether to use original or augmented dataset. Options: [`original`, `rem1`, `rem2`, `rem3`, `rem4`, `ori1`, `low1`]. Default: `original`")
    parser.add_argument('--scenario',  default='llm-based', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--train_epochs", default=15, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--output_dir',  default='experiments', type=str)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to train the model")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to eval the model")

    args = parser.parse_args()
    model_name = args.model_name_or_path.split('/')[-1]
    train_dataset = "" if args.train_dataset == 'original' else "-" + args.train_dataset
    output_folder = os.path.join(args.output_dir, args.scenario + train_dataset, f"{model_name}_{str(args.train_batch_size)}")
    print(f"current scenario - {output_folder}")
    if args.do_train:
        if os.path.exists(os.path.join(output_folder, f'evaluation_score_{args.seed}.json')):
            print('This scenario has been executed.')
            sys.exit(0)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    args.output_dir = output_folder

    if not args.do_train and args.do_eval:
        aug_eval = 'original'
        if str(args.eval_dataset).startswith('rem'):
            aug_eval = 'remove'
        elif str(args.eval_dataset).startswith('low'):
            aug_eval = 'lower'
        train_eval = args.train_dataset + '_' + aug_eval
        args.eval_output = os.path.join("experiments", 'evaluation', args.scenario, f"{model_name}_{str(args.train_epochs)}", train_eval, args.eval_dataset, str(args.seed))
        if os.path.exists(os.path.join(args.eval_output, f'evaluation_score_{args.seed}.json')):
            print('This scenario has been executed.')
            sys.exit(0)
        if not os.path.exists(args.eval_output):
            os.makedirs(args.eval_output)
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


def process_predictions(predictions: List[List[Dict]]) -> List[List[str]]:
    """Convert SimpleTransformers prediction format to list of labels."""
    processed_preds = []
    for sentence in predictions:
        # Each sentence is a list of dictionaries with one item
        labels = [list(word_dict.values())[0] for word_dict in sentence]
        processed_preds.append(labels)
    return processed_preds


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


def evaluate_predictions(true_sentences: List[List[str]], 
                        pred_labels: List[List[str]]) -> Dict:
    """Evaluate both boundary detection and sentence type classification."""
    
    all_true_spans = []
    all_pred_spans = []
    
    # Extract spans for all sentences
    for true_labels, pred_sentence_labels in zip(true_sentences, pred_labels):
        true_spans = extract_valid_spans(true_labels)
        pred_spans = extract_valid_spans(pred_sentence_labels)
        
        all_true_spans.append(true_spans)
        all_pred_spans.append(pred_spans)
    
    # Evaluate boundaries
    total_correct = 0
    total_correct_strict = 0
    total_predicted = 0
    total_true = 0
    
    # For classification evaluation
    true_types = []
    pred_types = []
    
    for true_spans, pred_spans in zip(all_true_spans, all_pred_spans):
        # Convert spans to boundary tuples
        true_boundaries = {(span.start_idx, span.end_idx) for span in true_spans}
        pred_boundaries = {(span.start_idx, span.end_idx) for span in pred_spans}
        true_boundaries_strict = {(span.start_idx, span.end_idx, span.entity_type) for span in true_spans}
        pred_boundaries_strict = {(span.start_idx, span.end_idx, span.entity_type) for span in pred_spans}

        # Get correctly identified boundaries
        correct_boundaries = true_boundaries.intersection(pred_boundaries)
        correct_boundaries_strict = true_boundaries_strict.intersection(pred_boundaries_strict)
        
        total_correct += len(correct_boundaries)
        total_correct_strict += len(correct_boundaries_strict)
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
    FP = total_predicted - total_correct_strict
    FN = total_true - total_correct_strict
    boundary_strict = {
        'precision': total_correct_strict / total_predicted if total_predicted > 0 else 0,
        'recall': total_correct_strict / total_true if total_true > 0 else 0,
        'num_correct': total_correct_strict,
        'num_predicted': total_predicted,
        'num_true': total_true,
        'FP': FP,
        'FN': FN
    }
    
    # Calculate F1 for boundary detection
    if boundary_metrics['precision'] + boundary_metrics['recall'] > 0:
        boundary_metrics['f1'] = 2 * (boundary_metrics['precision'] * boundary_metrics['recall']) / (boundary_metrics['precision'] + boundary_metrics['recall'])
    else:
        boundary_metrics['f1'] = 0
        
    if boundary_strict['precision'] + boundary_strict['recall'] > 0:
        boundary_strict['f1'] = 2 * (boundary_strict['precision'] * boundary_strict['recall']) / (boundary_strict['precision'] + boundary_strict['recall'])
    else:
        boundary_strict['f1'] = 0

    # Calculate classification metrics only for correctly identified boundaries
    if true_types:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_types, pred_types, average='binary', zero_division=0,
            labels=[1]  # Ensure we're calculating metrics for Event class
        )
        cm = confusion_matrix(true_types, pred_types, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        accuracy = accuracy_score(true_types, pred_types)
        spesificity = int(TN) / (int(TN) + int(FP)) if (int(TN) + int(FP)) > 0 else 0
        fp_rate = 1 - spesificity
        fn_rate = 1 - recall
        g_mean = math.sqrt(recall * spesificity)
        f1_abs = f1 * boundary_metrics['f1']
    else:
        precision = recall = f1 = accuracy = 0
        cm = confusion_matrix(true_types, pred_types, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        spesificity = fp_rate = fn_rate = 0
        g_mean = f1_abs = 0
    
    classification_metrics = {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
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
        f1_f1 = 2 * (boundary_metrics['f1'] * f1_abs) / (boundary_metrics['f1'] + f1_abs)
    else:
        f1_f1 = 0
    
    return {
        'boundary': boundary_metrics,
        'boundary_strict': boundary_strict,
        'classification': classification_metrics,
        'f1_f1': f1_f1
    }


# Example usage:
def evaluate_model(true_labels, pred_labels):
    # Evaluate
    metrics = evaluate_predictions(true_labels, pred_labels)
    
    print("\nBoundary Detection Metrics:")
    print(f"Precision: {metrics['boundary']['precision']:.4f}")
    print(f"Recall: {metrics['boundary']['recall']:.4f}")
    print(f"F1: {metrics['boundary']['f1']:.4f}")
    print(f"Correct Boundaries: {metrics['boundary']['num_correct']}")
    print(f"Predicted Boundaries: {metrics['boundary']['num_predicted']}")
    print(f"True Boundaries: {metrics['boundary']['num_true']}")

    print("\nStrict Boundary Detection Metrics:")
    print(f"Precision: {metrics['boundary_strict']['precision']:.4f}")
    print(f"Recall: {metrics['boundary_strict']['recall']:.4f}")
    print(f"F1: {metrics['boundary_strict']['f1']:.4f}")
    print(f"Correct Boundaries: {metrics['boundary_strict']['num_correct']}")
    print(f"Predicted Boundaries: {metrics['boundary_strict']['num_predicted']}")
    print(f"True Boundaries: {metrics['boundary_strict']['num_true']}")
    
    print("\nClassification Metrics (for correct boundaries):")
    print(f"Precision: {metrics['classification']['precision']:.4f}")
    print(f"Recall: {metrics['classification']['recall']:.4f}")
    print(f"F1: {metrics['classification']['f1']:.4f}")
    print(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
    print(f"Support (correct boundaries): {metrics['classification']['support']}")
    
    return metrics


def log_predictions_to_excel(args, true_sentences: List[Tuple[List[str], List[str]]], 
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
            os.path.join(output_dir, f'correct_predictions_{args.seed}.xlsx'),
            index=False
        )
    
    if incorrect_boundaries_data:
        incorrect_df = pd.DataFrame(incorrect_boundaries_data)
        incorrect_df.to_excel(
            os.path.join(output_dir, f'incorrect_predictions_{args.seed}.xlsx'),
            index=False
        )
    
    print(f"Logged {len(correct_boundaries_data)} correct predictions")
    print(f"Logged {len(incorrect_boundaries_data)} incorrect predictions")
    print(f"Files saved in directory: {output_dir}")


def log_word_level_predictions(args, true_sentences: List[Tuple[List[str], List[str]]], 
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
    with pd.ExcelWriter(os.path.join(output_dir, f'word_level_predictions_{args.seed}.xlsx'),
                       engine='openpyxl') as writer:
        styled_df.to_excel(writer, index=False)
    
    # Create summary of error types
    error_summary = pd.DataFrame(word_level_df[~word_level_df['is_boundary_correct']]['error_type'].value_counts())
    error_summary.columns = ['count']
    error_summary.to_excel(os.path.join(output_dir, f'word_level_error_type_summary_{args.seed}.xlsx'))
    
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
        

def main():
    # initialization
    args = init_args()
    seed_everything(args.seed)
    # Read data
    if args.do_train:
        train_path = os.path.join("dataset", "train_conll_data.txt")
        test_path = os.path.join("dataset", "test_conll_data.txt")
        if str(args.train_dataset).startswith("aug"):
            filename = "train_augmented_" + str(args.train_dataset).split('-')[-1] + ".txt"
            train_path = os.path.join("dataset", filename)
        elif str(args.train_dataset).startswith("rem"):
            filename = "train_augmented_remove_" + str(args.train_dataset).split('-')[-1] + ".txt"
            train_path = os.path.join("dataset", filename)
        elif str(args.train_dataset).startswith("fold"):
            fold = str(args.train_dataset).split("-")[-1]
            train_path = os.path.join("dataset", "cross-validation", f"train_{fold}.txt")
            test_path = os.path.join("dataset", "cross-validation", f"test_{fold}.txt")
        elif str(args.train_dataset).startswith("seed"):
            seed = str(args.train_dataset).split("-")[-1]
            train_path = os.path.join("dataset", "multiple-split", f"train_{seed}.txt")
            test_path = os.path.join("dataset", "multiple-split", f"test_{seed}.txt")

        val_sentences = read_conll_file(test_path)

        # Get unique labels
        labels = ['O',
                'B-Event', 'I-Event', 'E-Event', 'S-Event',
                'B-NonEvent', 'I-NonEvent', 'E-NonEvent', 'S-NonEvent',
                ]
        
        # Define model arguments
        model_args = {
            'num_train_epochs': args.train_epochs,
            'learning_rate': args.learning_rate,
            'overwrite_output_dir': True,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size,
            'output_dir': args.output_dir,
            'save_steps': -1,
            'save_model_every_epoch': False
        }

        if 'uncased' in args.model_name_or_path:
            model_args['do_lower_case'] = True
        # Initialize model (can use any transformer model)
        model = NERModel(
            args.model_type,        # Model type (can be roberta, xlnet, etc.)
            args.model_name_or_path,    # Model name
            labels=labels,
            args=model_args
        )
        
        # Train the model
        model.train_model(train_path, show_running_loss=True)
    
    if args.do_eval:
        # Make predictions on test set
        # Get predictions
        output_dir = args.output_dir
        if not args.do_train:
            model = NERModel(
                args.model_type,
                args.output_dir
            )

            if args.eval_dataset == 'original':
                eval_path = os.path.join("dataset", "test_conll_data.txt")
            elif str(args.eval_dataset).startswith('rem'):
                sg = args.eval_dataset[-1]
                eval_path = os.path.join("dataset", "sensitivity", "remove", f"sg{sg}_remove.txt")
            elif str(args.eval_dataset).startswith('low'):
                sg = args.eval_dataset[-1]
                eval_path = os.path.join("dataset", "sensitivity", "lower", f"sg{sg}_lower.txt")
            elif str(args.eval_dataset).startswith('ori'):
                sg = args.eval_dataset[-1]
                eval_path = os.path.join("dataset", "sensitivity", "original", f"sg{sg}_original.txt")
            elif str(args.train_dataset).startswith("fold"):
                fold = str(args.train_dataset).split("-")[-1]
                eval_path = os.path.join("dataset", "cross-validation", f"test_{fold}.txt")
            elif str(args.train_dataset).startswith("seed"):
                seed = str(args.train_dataset).split("-")[-1]
                eval_path = os.path.join("dataset", "multiple-split", f"test_{seed}.txt")
            val_sentences = read_conll_file(eval_path)
            output_dir = args.eval_output

        predictions, raw_preds = model.predict([words for words, _ in val_sentences], split_on_space=False)

        # Process predictions to get labels
        pred_labels = process_predictions(predictions)

        # Get true labels
        true_labels = [labels for _, labels in val_sentences]
        metrics = evaluate_model(true_labels, pred_labels)

        with open(os.path.join(output_dir, f"raw_preds_{args.seed}.json"), "w") as f:
            json.dump(eval(str(raw_preds)), f, indent=4)
        with open(os.path.join(output_dir, f"evaluation_score_{args.seed}.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(output_dir, f"true_labels_{args.seed}.json"), "w") as f:
            json.dump(true_labels, f, indent=4)
        with open(os.path.join(output_dir, f"pred_labels_{args.seed}.json"), "w") as f:
            json.dump(pred_labels, f, indent=4)

        # Log predictions for error analysis
        log_predictions_to_excel(
            args,
            true_sentences=val_sentences,
            pred_labels=pred_labels,
            output_dir=output_dir
        )

        # Log word-level predictions for detailed error analysis
        log_word_level_predictions(
            args,
            true_sentences=val_sentences,
            pred_labels=pred_labels,
            output_dir=output_dir
        )
    
    sys.exit(0)

if __name__ == '__main__':
    main()