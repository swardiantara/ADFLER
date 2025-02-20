import json
import os
import numpy as np

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, AutoTokenizer

from finetune import init_args, seed_everything
from src.token_classification import TokenClassificationExplainer


def handle_bert(word_attributions):
    tokens = []
    attribution_matrix = []
    predicted_tags = []
    indices_to_remove = []
    for i, item in enumerate(word_attributions):
        scores = [score[1] for score in item['attribution_scores']]
        attribution_matrix.append(scores)
        if item['token'] not in ['[CLS]', '[SEP]']:
            if not '##' in item['token']:
                tokens.append(item['token'])
                predicted_tags.append(item['label'])
            else:
                # if '##' in item['token']:
                indices_to_remove.append(i)
                tokens[-1] = tokens[-1] + item['token'][2:]
                # if 'Ġ' in item['token']:
                #     indices_to_remove.append(i)
                #     tokens[-1] = tokens[-1] + item['token'][1:]
        else:
             indices_to_remove.append(i)
    return tokens, np.array(attribution_matrix), predicted_tags, indices_to_remove


def handle_roberta(word_attributions):
    tokens = []
    attribution_matrix = []
    predicted_tags = []
    indices_to_remove = []
    for i, item in enumerate(word_attributions):
        scores = [score[1] for score in item['attribution_scores']]
        attribution_matrix.append(scores)
        if item['token'] not in ['<s>', '</s>']:
            if i == 0:
                tokens.append(item['token'])
                predicted_tags.append(item['label'])
            if 'Ġ' in item['token']:
                tokens.append(item['token'][1:])
                predicted_tags.append(item['label'])
            else:
                indices_to_remove.append(i)
                tokens[-1] = tokens[-1] + item['token'][1:]
        else:
             indices_to_remove.append(i)
    return tokens, np.array(attribution_matrix), predicted_tags, indices_to_remove


def handle_xlnet(word_attributions):
    tokens = []
    attribution_matrix = []
    predicted_tags = []
    indices_to_remove = []
    for i, item in enumerate(word_attributions):
        scores = [score[1] for score in item['attribution_scores']]
        attribution_matrix.append(scores)
        if '_' in item['token']:
            predicted_tags.append(item['label'])
        else:
            tokens.append(item['token'][1:])
            indices_to_remove.append(i)
            tokens[-1] = tokens[-1] + item['token']

    return tokens, np.array(attribution_matrix), predicted_tags, indices_to_remove


def handle_albert(word_attributions):
    tokens = []
    attribution_matrix = []
    predicted_tags = []
    indices_to_remove = []
    for i, item in enumerate(word_attributions):
        scores = [score[1] for score in item['attribution_scores']]
        attribution_matrix.append(scores)
        if '_' in item['token']:
            predicted_tags.append(item['label'])
            tokens.append(item['token'][1:])
        else:
            indices_to_remove.append(i)
            tokens[-1] = tokens[-1] + item['token']

    return tokens, np.array(attribution_matrix), predicted_tags, indices_to_remove


def create_heatmap(word_attributions, model_name: str, filename: str):
    handlers = {
        "bert-base-cased": handle_bert,
        "bert-base-uncased": handle_bert,
        "distilbert-base-cased": handle_bert,
        "distilbert-base-uncased": handle_bert,
        "roberta-base": handle_roberta,
        "distilroberta-base": handle_roberta,
        "xlnet-base-cased": handle_xlnet,
        # "albert-base-v2": handle_albert
    }
    # extract tokens and their attribution scores
    tokens = []
    attribution_matrix = []
    predicted_tags = []
    indices_to_remove = []
    if model_name != 'xlnet-base-cased':
        # choose handler based on model_type 
        handler = handlers.get(model_name, lambda: print(f"The model {model_name} is not supported"))
        tokens, attribution_matrix, predicted_tags, indices_to_remove = handler(word_attributions)
    else:
        # electra-base-discriminator handler
        for i, item in enumerate(word_attributions):
            scores = [score[1] for score in item['attribution_scores']]
            attribution_matrix.append(scores)
            tokens.append(item['token'])
            predicted_tags.append(item['label'])
            attribution_matrix = np.array(attribution_matrix)

    rows_to_keep = list(set(range(len(word_attributions))) - set(indices_to_remove))
    reduced_matrix = attribution_matrix[np.ix_(rows_to_keep, rows_to_keep)]
    # transpose for easier interpretability
    scores_matrix = reduced_matrix.transpose()

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
    args = init_args()
    seed_everything(args.seed)

    # model_paths = [
    #      os.path.join('experiments', 'llm-seed', 'bert-base-cased_15'),         # bert-base-cased trained on original
    #      os.path.join('experiments', 'capital', 'bert-base-uncased_15'),        # bert-base-uncased trained on original
    #      os.path.join('experiments', 'llm-seed-rem-100', 'bert-base-cased_15'), # bert-base-cased trained on rem-100
    #      os.path.join('experiments', 'capital-rem-100', 'bert-base-uncased_15'),# bert-base-uncased trained on rem-100
    #      os.path.join('experiments', 'llm-seed', 'roberta-base_15'),            # roberta-base trained on original
    #      os.path.join('experiments', 'llm-seed-rem-100', 'roberta-base_15'),    # roberta-base trained on rem-100
    # ]
    # for model_path in tqdm(model_paths):
    # output_dir = os.path.join('visualization', 'interpretability', model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

    output_dir = os.path.join(args.output_dir, f"interpretability_{args.seed}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
    model_name = args.model_name_or_path.split('/')[-1]
    for idx, sample in tqdm(enumerate(samples), desc="Generating word importance heatmap..."):
        filename = os.path.join(output_dir, f'sample_{idx}.pdf')
        if os.path.exists(filename):
            print('sample_{idx} has been generated.')
            continue
        word_attributions = ner_explainer(sample)
        create_heatmap(word_attributions, model_name, filename)
        # html = ner_explainer.visualize()
        # soup = BeautifulSoup(html.data, "html.parser")
        with open(os.path.join(output_dir, f"sample_{idx}.json"), "w") as f:
                json.dump(word_attributions, f, indent=4)
        # with open(os.path.join(output_dir, f"sample_{idx}.html"), "w", encoding = 'utf-8') as f:
        #         f.write(str(soup.prettify()))


if __name__ == "__main__":
    main()