import glob
import pandas as pd
import argparse
from tqdm import tqdm
import os

"""
Script to switch labels
"""

if __name__ == "__main__":
    print("Starting...")
    default_output_folder = '/root/datasets/siwm/'
    default_output_path = default_output_folder + 'labels_inverted.csv'

    if not os.path.isdir(default_output_folder):
        os.makedirs(default_output_folder)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", required=True, help="Absolute path to labels csv.")
    parser.add_argument("-o", "--output_csv", required=False, default=default_output_path,
                        help="Absolute path to output csv.")
    args = parser.parse_args()

    args.file_path = args.file_path.rstrip("/")

    print("Processing file ", args.file_path)
    
    df = pd.read_csv(args.file_path)

    zeros_indices = df.loc[df['target'] == 0].index.values
    ones_indices = df.loc[df['target'] == 1].index.values

    df.loc[zeros_indices, 'target'] = 1
    df.loc[ones_indices, 'target'] = 0

    df.to_csv(args.output_csv, index=False)
