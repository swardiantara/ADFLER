# def extract_boundaries_with_types(tags):
#     """
#     Extract sentence boundaries with entity types (Event or NonEvent) using BIOES tagging.
#     """
#     boundaries = []
#     start_idx = None
#     entity_type = None
#     for idx, tag in enumerate(tags):
#         if tag.startswith("B-"):
#             start_idx = idx
#             entity_type = tag[2:]
#         elif tag.startswith("E-") and start_idx is not None and entity_type == tag[2:]:
#             end_idx = idx
#             boundaries.append((start_idx, end_idx, entity_type))
#             start_idx = None
#             entity_type = None
#         elif tag.startswith("S-"):
#             single_idx = idx
#             entity_type = tag[2:]
#             boundaries.append((single_idx, single_idx, entity_type))
#         elif tag.startswith("I-") and start_idx is not None and entity_type != tag[2:]:
#             print('Intermediary tag is not the same!')
#     return boundaries
def extract_boundaries_with_types(tags):
    """
    Extracts boundaries with entity types based on specified rules for sentence boundaries.
    
    Args:
    - tags: List of BIOES tags for a single input sample.
    
    Returns:
    - boundaries: List of tuples (start_idx, end_idx, entity_type) representing each boundary.
    """
    boundaries = []
    start_idx = None
    current_entity_type = None

    for idx, tag in enumerate(tags):
        if tag == "O":  # Separator tag; close the current boundary if one is open
            if start_idx is not None:
                boundaries.append((start_idx, idx - 1, current_entity_type))
                start_idx = None
                current_entity_type = None
            continue

        tag_parts = tag.split("-")
        if len(tag_parts) == 2:
            prefix, entity_type = tag_parts
        else:
            # Handle cases where the tag might not follow the "B-", "I-", "E-", "S-" structure
            continue

        if start_idx is None:
            # Begin a new boundary
            start_idx = idx
            current_entity_type = entity_type

        elif entity_type != current_entity_type:
            # Different entity type encountered, close current boundary and start new
            boundaries.append((start_idx, idx - 1, current_entity_type))
            start_idx = idx
            current_entity_type = entity_type

        elif (prefix == "B") or (prefix == "S"):
            # B or S tag appears; start new boundary
            if start_idx is not None:
                boundaries.append((start_idx, idx - 1, current_entity_type))
            start_idx = idx
            current_entity_type = entity_type

    # Close any remaining boundary
    if start_idx is not None:
        boundaries.append((start_idx, len(tags) - 1, current_entity_type))

    return boundaries


def evaluate_sbd_boundary_only(all_true_tags, all_pred_tags):
    """
    Evaluate sentence boundary disambiguation (SBD) without considering entity types.
    """
    tp, fp, fn = 0, 0, 0
    
    for true_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        true_boundaries = {(start, end) for start, end, _ in extract_boundaries_with_types(true_tags)}
        pred_boundaries = {(start, end) for start, end, _ in extract_boundaries_with_types(pred_tags)}

        tp += len(true_boundaries & pred_boundaries)  # Correct boundaries
        fp += len(pred_boundaries - true_boundaries)  # Extra boundaries
        fn += len(true_boundaries - pred_boundaries)  # Missing boundaries

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn
    }


def evaluate_classification_correct_boundaries(all_true_tags, all_pred_tags):
    """
    Evaluate binary classification (Event vs. NonEvent) on correctly segmented boundaries.
    Treat "Event" as positive and "NonEvent" as negative class.
    Only include sentences with correctly identified boundaries.
    """
    tp, fp, fn = 0, 0, 0
    correct_boundary_count = 0  # Support based on correct boundaries only

    for true_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        true_boundaries = [(start, end) for start, end, _ in extract_boundaries_with_types(true_tags)]
        pred_boundaries = [(start, end) for start, end, _ in extract_boundaries_with_types(pred_tags)]
        true_boundaries_types = extract_boundaries_with_types(true_tags)
        pred_boundaries_types = extract_boundaries_with_types(pred_tags)

        # Identify correctly segmented boundaries
        correct_boundaries = set(true_boundaries).intersection(set(pred_boundaries))
        # Increment the correct boundary count for each correct boundary
        correct_boundary_count += len(correct_boundaries)
        
        for boundary in correct_boundaries:
            _, _, true_type = true_boundaries_types[true_boundaries.index(boundary)]
            _, _, pred_type = pred_boundaries_types[pred_boundaries.index(boundary)]
            if true_type == pred_type:
                # Correct boundary with correct entity type (True Positive for both Event and NonEvent)
                tp += 1
            elif true_type == "NonEvent" and pred_type == "Event":
                # Incorrectly predicted as Event (False Positive)
                fp += 1
            elif true_type == "Event" and pred_type == "NonEvent":
                # Incorrectly predicted as NonEvent (False Negative)
                fn += 1

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": correct_boundary_count
    }


def log_errors_for_analysis(all_pred_tags, all_true_tags, all_tokens):
    """
    Logs segmentation and classification errors for error analysis, including aligned raw input words.
    
    Args:
    - all_true_tags: List of lists containing true BIOES tags for each sample.
    - all_pred_tags: List of lists containing predicted BIOES tags for each sample.
    - all_tokens: List of tokenized words from BERT.
    
    Returns:
    - logs: Dictionary with details of segmentation and classification errors.
    """
    logs = {
        "segmentation_correct": [],
        "segmentation_errors": [],
        "classification_correct": [],
        "classification_errors": []
    }

    for idx, (pred_tags, true_tags, words) in enumerate(zip(all_pred_tags, all_true_tags, all_tokens)):
        # Extract words from raw input tuples
        
        # Extract boundaries with entity types
        true_boundaries = [(start, end) for start, end, _ in extract_boundaries_with_types(true_tags)]
        pred_boundaries = [(start, end) for start, end, _ in extract_boundaries_with_types(pred_tags)]
        true_boundaries_types = extract_boundaries_with_types(true_tags)
        pred_boundaries_types = extract_boundaries_with_types(pred_tags)
        
        # Identify correctly segmented boundaries
        correct_boundaries = set(true_boundaries).intersection(set(pred_boundaries))
        incorrect_boundaries = set(pred_boundaries) - correct_boundaries
        
        # Log segmentation correct if there are no incorrect boundaries
        if len(incorrect_boundaries) == 0:
            logs["segmentation_correct"].append({
                "sample_index": idx,
                "raw_input": words,
                "true_tags": true_tags,
                "pred_tags": pred_tags,
                "true_boundaries": true_boundaries,
                "predicted_boundaries": pred_boundaries,
            })

        # Log segmentation errors if there are incorrect boundaries
        if len(incorrect_boundaries) > 0:
            logs["segmentation_errors"].append({
                "sample_index": idx,
                "raw_input": words,
                "true_tags": true_tags,
                "pred_tags": pred_tags,
                "true_boundaries": true_boundaries,
                "predicted_boundaries": pred_boundaries,
                "incorrect_boundaries": list(incorrect_boundaries)
            })

        # Log classification results for correctly segmented boundaries
        for boundary in correct_boundaries:
            start, end, true_type = true_boundaries_types[true_boundaries.index(boundary)]
            _, _, pred_type = pred_boundaries_types[pred_boundaries.index(boundary)]

            log_entry = {
                "sample_index": idx,
                "raw_input": words[start:end+1],
                "boundary": (start, end),
                "true_type": true_type,
                "pred_type": pred_type,
                "true_tags": true_tags[start:end+1],
                "pred_tags": pred_tags[start:end+1]
            }

            if true_type == pred_type:
                # Correct classification
                logs["classification_correct"].append(log_entry)
            else:
                # Classification error within correct boundaries
                logs["classification_errors"].append(log_entry)

    return logs