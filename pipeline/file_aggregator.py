import os
import pandas as pd

directory_path = '../data'
all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

all_files_v = []
all_files_t = []

for file in all_files:
    if file.endswith('_validator_transactions.csv'):
        all_files_v.append(file)
    elif file.endswith('_transactions.csv'):
        all_files_t.append(file)

## Post-pipeline data aggregator.
def do_extraction():
    fail_list = []

    for file in all_files_v + all_files_t:
        try:
            _df = pd.read_csv(file)
            print(f"SUCESS: {file}")
        except Exception as e:
            fail_list.append(file)
            print(f"FAIL: {file} + {e}")

    directory_path = './data'
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for file in all_files:
        if file.endswith('.csv'):
            clean_csv(file)

    validator_files = [file for file in all_files if file.endswith('_validator_transactions.csv')]
    transaction_files = [file for file in all_files if file.endswith('_transactions.csv') and not file.endswith('_validator_transactions.csv')]

    for fail in fail_list:
        print(f"Failed to load file: {fail}")

    if validator_files:
        validators_combined = aggregate_files(validator_files)
        validators_combined.to_csv('./data/aggregated_validators.csv', index=False)

    if transaction_files:
        transactions_combined = aggregate_files(transaction_files)
        transactions_combined.to_csv('./data/aggregated_transactions.csv', index=False)

def clean_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip().split(',')
    num_headers = len(header)
    print(f"Processing file: {file_path}")
    print(f"Number of headers: {num_headers}")

    cleaned_lines = [lines[0]]  # Keep the header line
    for i, line in enumerate(lines[1:], start=2):
        columns = line.strip().split(',')
        if len(columns) > num_headers:
            print(f"Line {i} has {len(columns)} columns, cutting to {num_headers} columns.")
            columns = columns[:num_headers]
        cleaned_lines.append(','.join(columns) + '\n')

    with open(file_path, 'w') as file:
        file.writelines(cleaned_lines)

def aggregate_files(file_list):
    dataframes = [pd.read_csv(file) for file in file_list]
    return pd.concat(dataframes, ignore_index=True)