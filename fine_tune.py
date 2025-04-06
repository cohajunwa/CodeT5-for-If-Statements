import argparse
import os
import re
import sys

from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset


def flatten(example):
  method = example['cleaned_method']

  flattened_method = method.replace('    ', '<TAB>').replace('\n', '')
  example['cleaned_method'] = flattened_method
  return example

def mask(example):
  if_statement_pattern = r"if\s+.*"

  masked_method = re.sub(if_statement_pattern, "<IF-STMT>",
                         example['cleaned_method'], 1)
  example['masked_method'] = masked_method

  return example

def tokenization(examples, tokenizer):
    """Tokenize dataset using pre-trained tokenizer"""
    
    inputs = examples["masked_method"]
    targets = examples["target_block"]
    model_inputs = tokenizer(inputs, max_length=256, truncation = True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def run(dataset):
    """Pipeline for loading pre-trained Code-T5 model and tokenizer, modifying dataset, and fine-tuning"""
    
    print("Loading model and tokenizer")

    model_checkpoint = "Salesforce/codet5-small"

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

    tokenizer.add_tokens(["<IF-STMT>", "<TAB>"])

    model.resize_token_embeddings(len(tokenizer))


    print("Modifying dataset by flattening methods and masking if conditions")    
    dataset = dataset.map(flatten).map(mask)
    print(dataset)

    print("Tokenizing dataset")
    dataset = dataset.map(tokenization, fn_kwargs={"tokenizer": tokenizer})
    print(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type = str, help = "File containing Python functions for training")
    parser.add_argument("--validation_file", type = str, help = "File containing Python functions for validation")
    parser.add_argument("--test_file", type = str, help = "File containing Python functions for testing")

    args = parser.parse_args()

    if not os.path.isfile(args.train_file) or not os.path.isfile(args.validation_file) or not os.path.isfile(args.test_file):
        print("Missing file path(s) for training, validation, and/or testing dataset")
        sys.exit()
    
    print("Loading dataset for fine-tuning CodeT5")
    dataset = load_dataset("csv", data_files = {"train": args.train_file, "validation": args.validation_file, "test": args.test_file})

    print(dataset)
    run(dataset)
