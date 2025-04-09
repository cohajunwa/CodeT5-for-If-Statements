import argparse
import ast
import csv
import os
import pandas as pd
import re
import sys
import warnings

from pydriller import Repository
from pygments.lexers.python import PythonLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

IF_STATEMENT_PATTERN = r"if\s+.*?:"
warnings.filterwarnings("ignore", category = SyntaxWarning) # Prevent SyntaxWarnings from appearing during parsing

def extract_methods_from_python(code):
    """
    Extract methods from Python code.

    Args:
        code (str): Python code.

    Returns:
        list: List of extracted methods.
    """
    methods = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Remove docstrings
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                     node.body = node.body[1:]

                method_code = ast.unparse(node)
                methods.append(method_code)
    except SyntaxError:
        pass

    return methods

def extract_methods_to_dataframe_from_master(repo_paths):
    """
    Extract methods from Python files in the master branch and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
    """
    extracted_methods = []

    for repo_path in repo_paths:
        print(f"Processing repository: {repo_path}")
        try:
            for commit in Repository(repo_path).traverse_commits():
                #We only look into the modified files. In other words, we are looking into the history of the software system by traversing each commit.
                #Various Generative AI methods for SD have been trained on data collected in this way; for example bug fixing.
                for modified_file in commit.modified_files:
                    if modified_file.filename.endswith(".py") and modified_file.source_code:
                        methods = extract_methods_from_python(modified_file.source_code)

                        for method_code in methods:
                            extracted_methods.append({"Method Code": method_code})
        except Exception as e:
            print(f"Skipping {repo_path} due to issues that occurred during parsing") 
            continue # Skipping problematic repos

    df = pd.DataFrame(extracted_methods)
    return df

def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from Python methods in a DataFrame and adds a new column with cleaned methods.

    Args:
        df (pd.DataFrame): DataFrame containing the methods.
        method_column (str): Column name containing the raw Java methods.
        language (str): Programming language for the lexer (e.g., 'python').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Method Code No Comments'.
    """
    # Define a function to remove comments from a single method
    def remove_comments(code):
        lexer = PythonLexer()
        tokens = lexer.get_tokens(code)
        # Filter out comments using a lambda function

        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))

        return clean_code

    # Apply the function to the specified column and add a new column with the results
    df["Method Code No Comments"] = df[method_column].apply(remove_comments)
    return df

def remove_duplicates(data):
    """Remove duplicate methods based on method content.
      Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Code")

def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    data = data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Code"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def create_raw_dataframe_from_repos(repo_list_file, save):
    repo_list = []
    with open(repo_list_file) as file:
        for line in file:
            repo_list.append(f"https://www.github.com/{line.rstrip()}")
    
    raw_df = extract_methods_to_dataframe_from_master(repo_list)

    if save:
        print("Saving raw dataframe to extracted_methods.csv")
        raw_df.to_csv("../extracted_methods.csv")
    return raw_df

def clean_methods_dataframe(raw_df, save):
    clean_df = remove_comments_from_dataframe(raw_df, "Method Code", "Python")
    clean_df = remove_duplicates(clean_df)
    clean_df = filter_ascii_methods(clean_df)
    clean_df = remove_outliers(clean_df)

    if save:
        print("Saving cleaned dataframe to cleaned_methods.csv")
        clean_df.to_csv("../cleaned_methods.csv")
    return clean_df

def format_dataset_for_llm(clean_df, output_data_file):
    lexer = PythonLexer()

    final_df = clean_df.drop(columns=["Method Code"])
    final_df = final_df.rename(columns={"Method Code No Comments": "cleaned_method"})

    final_df = final_df.loc[final_df['cleaned_method'].str.contains(IF_STATEMENT_PATTERN, regex=True)].copy()
    final_df['target_block'] = final_df['cleaned_method'].str.extract(f'({IF_STATEMENT_PATTERN})')

    final_df['tokens_in_method'] = final_df['cleaned_method'].apply(lambda code: len([t[1] for t in lexer.get_tokens(code)]))

    print(f"Saving final dataframe to {output_data_file}")
    final_df.to_csv(f"../{output_data_file}")

    return final_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--git_repo_file", type = str, default = '../repo_list.txt', help = 'File containing list of Git repos to generate dataset, each separated by newline')
    parser.add_argument("--save_intermediate_files", type = bool, default = False, help = 'Determine whether to save intermediate dataframes (original raw dataframe from extracted methods and preprocessed dataframe with cleaned methods)')
    parser.add_argument("--output_data_file", type = str, default = 'ft_train.csv', help = 'Prepared dataset for fine-tuning CodeT5 model (csv file)')
    args = parser.parse_args()

    # Ensuring Git repo file exists
    if not os.path.isfile(args.git_repo_file):
        print("Git repo file not found")
        sys.exit(1)

    # Ensuring output filename is valid
    if not args.output_data_file.endswith(".csv"):
        print(f"Error: Output file must be a `.csv` file.")
        sys.exit(1)
    
    print("Creating raw dataframe by extracting Python methods from Git repositories")
    raw_df = create_raw_dataframe_from_repos(args.git_repo_file, args.save_intermediate_files)
    print(f"Number of collected Python functions: {len(raw_df)}")

    print("Cleaning methods in raw dataframe")
    clean_df = clean_methods_dataframe(raw_df, args.save_intermediate_files)
    print(f"Number of Python functions after preprocessing: {len(clean_df)}")

    print("Preparing dataframe for CodeT5 models with columns 'cleaned_method', 'target_block', and 'tokens_in_method")
    final_df = format_dataset_for_llm(clean_df, args.output_data_file)
    print(f"Number of Python functions in the final dataframe: {len(final_df)}")

    