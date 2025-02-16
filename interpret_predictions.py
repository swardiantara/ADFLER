import json
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, AutoTokenizer
from bs4 import BeautifulSoup 

from src.token_classification import TokenClassificationExplainer


def create_heatmap(data, filename):
    # extract tokens and their attribution scores
    tokens = []
    attribution_matrix = []
    predicted_tags = []
    
    for item in data:
        # if token not in ['<s>', '</s>']:  # skip special tokens if needed
        tokens.append(item['token'])
        scores = [score[1] for score in item['attribution_scores']]
        attribution_matrix.append(scores)
        predicted_tags.append(item['label'])
    
    # transpose for easier interpretability
    scores_matrix = np.array(attribution_matrix).transpose()

    # create the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(scores_matrix, annot=True, center=0, cmap="rocket_r", fmt='.2f', xticklabels=tokens, yticklabels=tokens, cbar_kws={'label': 'Attribution Score'})

    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens, rotation=0)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(ax.get_xticks())
    ax_top.set_xticklabels(predicted_tags, rotation=45, ha="left")
    ax_top.set_xlabel("Predicted Tags") # predicted tags

    # axes labels
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Contributing Tokens")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    return None


def main():
    model_paths = [
         os.path.join('experiments', 'llm-seed', 'bert-base-cased_15'),         # bert-base-cased trained on original
         os.path.join('experiments', 'capital', 'bert-base-uncased_15'),        # bert-base-uncased trained on original
         os.path.join('experiments', 'llm-seed-rem-100', 'bert-base-cased_15'), # bert-base-cased trained on rem-100
         os.path.join('experiments', 'capital-rem-100', 'bert-base-uncased_15'),# bert-base-uncased trained on rem-100
         os.path.join('experiments', 'llm-seed', 'roberta-base_15'),            # roberta-base trained on original
         os.path.join('experiments', 'llm-seed-rem-100', 'roberta-base_15'),    # roberta-base trained on rem-100
    ]

    for model_path in model_paths:
        output_dir = os.path.join('visualization', 'interpretability', model_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        ner_explainer = TokenClassificationExplainer(
            model,
            tokenizer,
        )

        samples = [
            "Unknown Error, Cannot Takeoff. Contact DJI support.",
            "Battery cell broken, please replace the battery.",
            "Strong Interference. Fly with caution.",
            "Low power, please replace the battery.",
            "Compass error, calibration required.",
            "Unknown Error Cannot Takeoff Contact DJI support.",
            "Battery cell broken please replace the battery.",
            "Strong Interference Fly with caution.",
            "Low power please replace the battery.",
            "Compass error calibration required.",
            "unknown error cannot takeoff contact dji support.",
            "battery cell broken please replace the battery.",
            "strong interference fly with caution.",
            "low power please replace the battery.",
            "compass error calibration required.",
        ]

        for idx, sample in enumerate(samples):
            filename = os.path.join(output_dir, f'sample_{idx}.pdf')
            word_attributions = ner_explainer(sample)
            create_heatmap(word_attributions, filename)
            html = ner_explainer.visualize()
            soup = BeautifulSoup(html, "html.parser")
            with open(os.path.join(output_dir, f"sample_{idx}.json"), "w") as f:
                    json.dump(word_attributions, f, indent=4)
            with open(os.path.join(output_dir, f"sample_{idx}.html"), "w", encoding = 'utf-8') as f:
                    f.write(str(soup.prettify()))


if __name__ == "__main__":
    main()