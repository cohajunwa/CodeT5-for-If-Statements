from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from evaluator import calc_code_bleu

import argparse
import pandas as pd
import sacrebleu
import torch

def make_prediction(model, tokenizer, example):
    """Retrieves prediction from example input"""

    tokenized_input = {'input_ids': torch.tensor([example['input_ids']]).to(model.device)}
    tokenized_prediction = model.generate(**tokenized_input)
    predicted_code = tokenizer.decode(tokenized_prediction[0], skip_special_tokens=True)
    return predicted_code

def get_exact_match(expected_code, predicted_code):
    """Returns whether expected code and predicted code are exact matches"""

    return expected_code == predicted_code

def get_bleu_4_score(expected_code, predicted_code):
    """Returns BLEU-4 score for predicted code"""

    return sacrebleu.corpus_bleu([predicted_code], [[expected_code]]).score

def get_code_bleu_score(expected_code, predicted_code):
    """Returns CodeBLEU score for predicted code"""

    return calc_code_bleu.code_bleu([predicted_code], [[expected_code]], 'python')

def get_results(model, tokenizer, tokenized_dataset):
    testset_results = []

    for example in tokenized_dataset['test']:
        masked_method = example['masked_method']
        expected_if_condition = example['target_block']
        predicted_if_condition = make_prediction(model, tokenizer, example)
        exact_match = get_exact_match(expected_if_condition, predicted_if_condition)
        bleu_4_score = get_bleu_4_score(expected_if_condition, predicted_if_condition)
        code_bleu_score = 0

        testset_results.append(
            {
                'input_function': masked_method,
                'expected_if_condition': expected_if_condition,
                'predicted_if_condition': predicted_if_condition,
                'exact_match':exact_match,
                'CodeBLEU score': code_bleu_score,
                'BLEU-4 score': bleu_4_score,
            }
        )

    testset_results_df = pd.DataFrame(testset_results)
    return testset_results_df
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_model_path", type = str, help = "Directory containing saved fine-tuned model & tokenizer")
    parser.add_argument("--load_tokenized_ds_path", type = str, help = "Directory containing saved tokenized dataset")
    parser.add_argument("--output_csv_file", type = str, default = "testset-results.csv", help = "Filename for saving results")
    args = parser.parse_args()

    print("Loading model, tokenizer, and tokenized_dataset")
    model = T5ForConditionalGeneration.from_pretrained(args.load_model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.load_model_path)
    tokenized_dataset = load_from_disk(args.load_tokenized_ds_path)

    print("Computing metrics")
    testset_results_df = get_results(model, tokenizer, tokenized_dataset)

    print(f"Saving results in {args.output_csv_file}")
    testset_results_df.to_csv(args.output_csv_file)
    
