import pandas as pd
import numpy as np

import argparse
import os
import glob
from tqdm import tqdm


def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


if __name__ == "__main__":
    print("Starting...")

    parser = argparse.ArgumentParser(
        description='Combine all datasets into one csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('root_folder', type=str, help='Absolute path to root folder containing all datasets.')
    parser.add_argument('-de', '--debug', type=bool, help='Debug mode.', default=False)

    args = parser.parse_args()

    fasd_folder = args.root_folder.rstrip("/") + "/casia-fasd"
    oulu_folder = args.root_folder.rstrip("/") + "/oulu"
    siwm_folder = args.root_folder.rstrip("/") + "/siwm"
    surf_folder = args.root_folder.rstrip("/") + "/surf"

    fasd_csvs = [
        fasd_folder + "/train_pruned.csv",
        fasd_folder + "/test_pruned.csv",
    ]

    # use protocol 4 and user 6 as holdout
    oulu_csvs = [
        oulu_folder + "/Protocols/Protocol_4/train_6.csv",
        oulu_folder + "/Protocols/Protocol_4/dev_6.csv",
        oulu_folder + "/Protocols/Protocol_4/test_6.csv",
    ]

    siwm_csvs = [
        siwm_folder + "/train_labels_inverted.csv",
        siwm_folder + "/val_labels_inverted.csv",
        siwm_folder + "/test_labels_inverted.csv",
    ]

    surf_csvs = [
        surf_folder + "/train_pruned.csv",
        surf_folder + "/val_pruned.csv",
        surf_folder + "/test_pruned.csv",
    ]

    combined_train_df = pd.DataFrame()
    combined_val_df = pd.DataFrame()
    combined_test_df = pd.DataFrame()

    # fasd
    train_df = pd.read_csv(fasd_csvs[0])
    val_df = pd.read_csv(fasd_csvs[1]) 

    combined_train_df = combined_train_df.append(train_df, ignore_index=True)
    combined_val_df = combined_val_df.append(val_df, ignore_index=True)

    # oulu p4
    train_df = pd.read_csv(oulu_csvs[0])
    val_df = pd.read_csv(oulu_csvs[1])
    test_df = pd.read_csv(oulu_csvs[2])

    combined_train_df = combined_train_df.append(train_df, ignore_index=True)
    combined_val_df = combined_val_df.append(val_df, ignore_index=True)
    combined_test_df = combined_test_df.append(test_df, ignore_index=True)

    # siwm
    train_df = pd.read_csv(siwm_csvs[0])
    val_df = pd.read_csv(siwm_csvs[1])
    test_df = pd.read_csv(siwm_csvs[2])

    combined_train_df = combined_train_df.append(train_df, ignore_index=True)
    combined_val_df = combined_val_df.append(val_df, ignore_index=True)
    combined_test_df = combined_test_df.append(test_df, ignore_index=True)

    # surf
    train_df = pd.read_csv(surf_csvs[0])
    val_df = pd.read_csv(surf_csvs[1])
    test_df = pd.read_csv(surf_csvs[2])

    combined_train_df = combined_train_df.append(train_df, ignore_index=True)
    combined_val_df = combined_val_df.append(val_df, ignore_index=True)
    combined_test_df = combined_test_df.append(test_df, ignore_index=True)

    combined_folder = args.root_folder.rstrip("/") + "/combined"
    make_folder(combined_folder)

    combined_train_df.to_csv(combined_folder + "/train.csv", index=False)
    combined_val_df.to_csv(combined_folder + "/val.csv", index=False)
    combined_test_df.to_csv(combined_folder + "/test.csv", index=False)
