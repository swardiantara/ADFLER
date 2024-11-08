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
                        pred_labels: List[List[Dict]]) -> Dict:
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
    boundary_metrics = {
        'precision': total_correct / total_predicted if total_predicted > 0 else 0,
        'recall': total_correct / total_true if total_true > 0 else 0,
        'num_correct': total_correct,
        'num_predicted': total_predicted,
        'num_true': total_true
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
        accuracy = accuracy_score(true_types, pred_types)
    else:
        precision = recall = f1 = accuracy = 0
    
    classification_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'support': len(true_types)  # Number of correctly identified boundaries
    }
    
    return {
        'boundary': boundary_metrics,
        'classification': classification_metrics
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


def main():
    # Read data
    train_path = os.path.join("dataset", "train_conll_data.txt")
    test_path = os.path.join("dataset", "test_conll_data.txt")
    val_sentences = read_conll_file(test_path)
    
    # Get unique labels
    labels = ['O',
              'B-Event', 'I-Event', 'E-Event', 'S-Event',
              'B-NonEvent', 'I-NonEvent', 'E-NonEvent', 'S-NonEvent',
              ]
    
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
        'bert-base-cased',  # Model name
        labels=labels,
        args=model_args
    )
    
    # Train the model
    model.train_model(train_path, show_running_loss=True)
    
    # Make predictions on validation set
    # Get predictions
    predictions, _ = model.predict([words for words, _ in val_sentences], split_on_space=False)
    # Process predictions to get labels
    pred_labels = process_predictions(predictions)
    # Get true labels
    true_labels = [labels for _, labels in val_sentences]
    # result, model_outputs, wrong_preds = model.eval_model(test_path)
    # metrics = evaluate_predictions(val_sentences, predictions)
    # print("Validation Metrics:", metrics)
    # print("================")
    # print(result)
    # print("================")
    # Compute the evaluation metrics
    evaluate_model(true_labels, pred_labels)
    # Log predictions for error analysis
    log_predictions_to_excel(
        true_sentences=val_sentences,
        pred_labels=pred_labels,
        output_dir="simple-trans"
    )

if __name__ == '__main__':
    main()