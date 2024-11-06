import os
import json
import argparse
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader

from src.eval_utils import log_errors_for_analysis, evaluate_sbd_boundary_only, evaluate_classification_correct_boundaries
from src.data_utils import NERDataset, DroneLogDataset
from src.llm_fine_tune import DroneLogNER, label2id

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true',
                        help="Whether to run inference with trained checkpoints")
    
    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--train_epochs", default=5, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--align_label", type=str, default='padding', choices=['padding', 'bioes'],
                        help="How to align the label of wordpiece tokens. Options: `padding` or `bioes`. Default: `padding`")

    # training details
    parser.add_argument('--output_dir',  default='experiments', type=str)
    parser.add_argument('--scenario',  default='llm-based', type=str,
                        help="Folder to store the experimental results. Default: `llm-based`")
    parser.add_argument('--save_model', action='store_true',
                        help="Whether to save the best checkpoint.")

    args = parser.parse_args()

    # create output folder if needed
    output_folder = os.path.join("experiments", args.scenario, args.model_name_or_path, 'align' if args.align_label else 'padding', "unidirectional", str(args.train_epochs), str(args.seed))
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


# Example usage
def main():
    # initialization
    args = init_args()
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize the model
    print('Model initialization and dataset preparation...')
    ner_model = DroneLogNER(model_name=args.model_name_or_path, device=device)
    train_path = os.path.join('dataset', 'train_conll_data.txt')
    test_path = os.path.join('dataset', 'test_conll_data.txt')

    if args.do_train:
        # Train the model
        print("Start model training...")
        ner_model.train(
            args,
            train_path=train_path,
            val_path=test_path
        )
        print("Finish model training...")

    if args.do_direct_eval:
        print("Start model evaluation...")
        # Evaluation

        _, all_pred_tags, all_true_tags, all_tokens = ner_model.evaluate(args, test_path, label2id)
        sbd_scores = evaluate_sbd_boundary_only(all_true_tags, all_pred_tags)
        classification_scores = evaluate_classification_correct_boundaries(all_true_tags, all_pred_tags)
        arguments_dict = vars(args)
        eval_score = {
            "scenario_args": arguments_dict,
            "sbd_score": sbd_scores,
            "classification_score": classification_scores
        }
        with open(os.path.join(args.output_dir, "evaluation_score.json"), "w") as f:
            json.dump(eval_score, f, indent=4)

        logs = log_errors_for_analysis(all_pred_tags, all_true_tags, all_tokens)
        with open(os.path.join(args.output_dir, "error_analysis_logs.json"), "w") as f:
            json.dump(logs, f, indent=4)

    if args.do_inference:
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