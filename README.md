# CodeT5-for-If-Statements

* [1. Introduction](#1-introduction)  
* [2. Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
* [3. Dataset Preparation, Model Fine-Tuning, and Evaluation](#3-dataset-preparation-model-fine-tuning-and-evaluation)  
  * [3.1 Preparing Training Dataset](#31-preparing-training-dataset)  
  * [3.2 Fine-Tuning](#32-fine-tuning)  
  * [3.3 Evaluation](#33-evaluation)  
* [4. Report](#4-report)  
---

# **1. Introduction** 

CodeT5 is a Transformer model pre-trained for code understanding and generation tasks. For this project, I fine-tune a small version of CodeT5 from Hugging Face called `codet5-small` to predict missing if-conditions in Python functions. Given a Python function where an if-statement is masked with a special token, the model learns to predict the missing condition. 

The project includes the following components:
* Generating a clean training dataset
* Preparing the train, validation, and test datasets for fine-tuning
* Fine-tuning the model with the train and validation datasets
* Evaluating the model's predictions on the test set using exact match, BLEU-4, and CodeBLEU


---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/cohajunwa/CodeT5-for-If-Statements.git
```

(2) Navigate into the project repository:
```
~ $ cd CodeT5-for-If-Statements
~/CodeT5-for-If-Statements $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:
```
~/CodeT5-for-If-Statements $ python -m venv ./venv/
~/CodeT5-for-If-Statements $ source venv/bin/activate
(venv) ~/CodeT5-for-If-Statements $ 
```

For Windows:
```
~/CodeT5-for-If-Statements $ python -m venv ./venv/
~/CodeT5-for-If-Statements $ ./venv/Scripts/activate
```

To deactivate the virtual environment, use the command:
```
(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:
```shell
(venv) ~/CodeT5-for-If-Statements $ pip install -r requirements.txt
```

---

# **3. Dataset Preparation, Model Fine-Tuning, and Evaluation**

All scripts for dataset preparation (```create_dataset.py```), model fine-tuning (```fine_tune.py```), and evaluation (```evaluate.py```) are contained in the ```src``` directory. To run these scripts, first `cd` into `src`.

## **3.1 Preparing Training Dataset**

I wrote `create_dataset.py` for preparing datasets for fine-tuning the model. It reads a text file called ```git_repos.txt```, which consists of a list of GitHub repositories (each separated by a newline), pulls the source code, and extracts Python functions. It also performs preprocessing by removing comments, filtering out duplicate functions, excluding those with non-ASCII characters, and discarding outliers based on function length. Finally, it prepares a dataframe consisting of three columns:
* cleaned_method: Contains the cleaned Python function
* target_block: The first if-statement in each function
* tokens_in_method: The number of tokens in the function

By default, ```create_dataset.py``` saves the dataframe in a CSV file called ```ft_train.csv``` in the project root.


```
(venv) ~/CodeT5-for-If-Statements/src $ python create_dataset.py --help
usage: create_dataset.py [-h] [--git_repo_file GIT_REPO_FILE] [--save_intermediate_files SAVE_INTERMEDIATE_FILES]
                         [--output_data_file OUTPUT_DATA_FILE]

options:
  -h, --help            show this help message and exit
  --git_repo_file GIT_REPO_FILE
                        File containing list of Git repos to generate dataset, each separated by newline (default:
                        ../repo_list.txt)
  --save_intermediate_files SAVE_INTERMEDIATE_FILES
                        Determine whether to save intermediate dataframes (original raw dataframe from extracted
                        methods and preprocessed dataframe with cleaned methods) (default: False)
  --output_data_file OUTPUT_DATA_FILE
                        Prepared dataset for fine-tuning CodeT5 model (csv file) (default: ft_train.csv)
```

I used this script to generate a training dataset consisting of ~50,000 Python functions using the following command:  
`(venv) ~/CodeT5-for-If-Statements/src $ python create_dataset.py --output_data_file student_ft_train.csv`

This dataset is in the repository and it is called `student_ft_train.csv`.

## **3.2 Fine-Tuning**
The ```fine_tune.py``` script is responsible for loading the pre-trained CodeT5 model, masking if-conditions, and fine-tuning. It requires three arguments: the training dataset, the validation dataset, and the test dataset. For instance, if you wanted to use the provided training, validation, and test datasets, you run

```
(venv) ~/CodeT5-for-If-Statements/src $ python fine_tune.py --train_file ../student_ft_train.csv --validation_file ../ft_valid.csv --test_file ../ft_test.csv
```

By default, the model is trained for 7 epochs, but users can also modify the number of training epochs. After training, the best model, tokenizer, and tokenized dataset are stored locally. Note that HuggingFace's trainer also automatically saves model checkpoints and wandb logs (if you choose to track your model's training run with [Weights and Biases](https://wandb.ai/)).

```
(venv) ~/CodeT5-for-If-Statements/src $ python fine_tune.py --help
usage: fine_tune.py [-h] [--train_file TRAIN_FILE] [--validation_file VALIDATION_FILE] [--test_file TEST_FILE]
                    [--num_train_epochs NUM_TRAIN_EPOCHS] [--save_model_path SAVE_MODEL_PATH]
                    [--save_tokenized_ds_path SAVE_TOKENIZED_DS_PATH]

options:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        File containing Python functions for training (default: None)
  --validation_file VALIDATION_FILE
                        File containing Python functions for validation (default: None)
  --test_file TEST_FILE
                        File containing Python functions for testing (default: None)
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs (default: 3)
  --save_model_path SAVE_MODEL_PATH
                        Path to save best model and tokenizer. If None, it won't be saved (default:
                        ../codet5-finetuned-if-condition-final)
  --save_tokenized_ds_path SAVE_TOKENIZED_DS_PATH
                        Directory to save tokenized dataset (default: ../tokenized_dataset)
```

## **3.3 Evaluation**
Finally, the `evaluate.py` script evaluates fine-tuned models on the test set and computes the scores for exact match, BLEU-4, and CodeBLEU. Ideally, you would run `evaluate.py` after completing a run of `fine_tune.py`:

```
(venv) ~/CodeT5-for-If-Statements/src $ python evaluate.py --load_model_path ../codet5-finetuned-if-condition-final --load_tokenized_ds_path ../tokenized_dataset
```

The output is a CSV file containing exact match, BLEU-4, and CodeBLEU scores for each prediction generated on the test dataset.

```
(venv) ~/CodeT5-for-If-Statements/src $ python evaluate.py --help
usage: evaluate.py [-h] [--load_model_path LOAD_MODEL_PATH] [--load_tokenized_ds_path LOAD_TOKENIZED_DS_PATH]
                   [--output_csv_file_path OUTPUT_CSV_FILE_PATH]

options:
  -h, --help            show this help message and exit
  --load_model_path LOAD_MODEL_PATH
                        Directory containing saved fine-tuned model & tokenizer (default: None)
  --load_tokenized_ds_path LOAD_TOKENIZED_DS_PATH
                        Directory containing saved tokenized dataset (default: None)
  --output_csv_file_path OUTPUT_CSV_FILE_PATH
                        Path to save csv file containing results (default: ../testset-results.csv)
```

My evaluation results for my fine-tuned model is found in `testset-results.csv`.

---


# **4. Report**
The assignment report is available in the file **GenAI for Software Development - Assignment 2.pdf**.
