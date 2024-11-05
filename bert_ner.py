import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

from src.eval_utils import log_errors_for_analysis, evaluate_sbd_boundary_only, evaluate_classification_correct_boundaries
from src.data_utils import NERDataset
from src.llm_fine_tune import DroneLogNER, label2id

# Example usage
def main():
    # Initialize the model
    print('Model initialization and dataset preparation...')
    ner_model = DroneLogNER(device='cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join('dataset', 'train_conll_data.txt')
    test_path = os.path.join('dataset', 'test_conll_data.txt')

    # Train the model
    print("Start model training...")
    ner_model.train(
        train_path=train_path,
        val_path=test_path,
        epochs=5
    )

    print("Start model evaluation...")
    # Evaluation
    val_dataset = NERDataset(test_path, ner_model.tokenizer, label_to_id=label2id)
    
    val_loader = DataLoader(val_dataset, batch_size=16)
    _, all_pred_tags, all_true_tags, all_tokens = ner_model.evaluate(val_loader)
    sbd_scores = evaluate_sbd_boundary_only(all_true_tags, all_pred_tags)
    classification_scores = evaluate_classification_correct_boundaries(all_true_tags, all_pred_tags)
    eval_score = {
        "sbd_score": sbd_scores,
        "classification_score": classification_scores
    }
    with open("evaluation_score.json", "w") as f:
        json.dump(eval_score, f, indent=4)

    logs = log_errors_for_analysis(all_pred_tags, all_true_tags, all_tokens)
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