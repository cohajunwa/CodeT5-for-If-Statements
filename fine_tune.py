import os
import sys

from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset


def modify_dataset(dataset):
    """Modify dataset by flattening methods and masking if conditions"""
    pass

def tokenization(dataset):
    """Tokenize dataset using pre-trained tokenizer"""
    pass

def load_model_and_tokenizer():
    """Load pre-trained Code-T5 model and tokenizer"""
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type = str, help = "File containing Python functions for training")
    parser.add_argument("--validation_file", type = str, help = "File containing Python functions for validation")
    parser.add_argument("--test_file", type = str, help = "File containing Python functions for testing")

    args = parser.parse_args()

    if not os.path.isfile(args.train_file) or not os.path.isfile(args.validation_file) or not os.path.isfile(args.test_file):
        print("Missing file path(s) for training, validation, and/or testing dataset")
        sys.exit()
    

    dataset = load_dataset("csv", data_files = {"train": args.train_file, "validation": args.validation_file, "test": args.test_file})
