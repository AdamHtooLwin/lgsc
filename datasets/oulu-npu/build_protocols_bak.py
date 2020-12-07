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
        description='Extract frames from OULU videos. Files should be named train/dev/test.csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('root_folder', type=str, help='Absolute path to root folder of dataset.')
    parser.add_argument('-de', '--debug', type=bool, help='Debug mode.', default=False)

    args = parser.parse_args()

    # load csvs
    train_csv = "/root/datasets/oulu/train.csv"
    dev_csv = "/root/datasets/oulu/dev.csv"
    test_csv = "/root/datasets/oulu/test.csv"

    train_folder = args.root_folder.rstrip("/") + "/Train_files/imgs"
    dev_folder = args.root_folder.rstrip("/") + "/Dev_files/imgs"
    test_folder = args.root_folder.rstrip("/") + "/Test_files/imgs"

    protocol_root_folder = args.root_folder.rstrip("/") + "/Protocols"
    protocols_folder = [
        protocol_root_folder + "/Protocol_1",
        protocol_root_folder + "/Protocol_2",
        protocol_root_folder + "/Protocol_3",
        protocol_root_folder + "/Protocol_4",
    ]

    make_folder(protocol_root_folder)
    for folder in protocols_folder:
        make_folder(folder)

    # protocol 1
    p1_train_df = pd.read_csv(protocols_folder[0] + '/Train.txt', sep=',', names=['label', 'video'])
    p1_dev_df = pd.read_csv(protocols_folder[0] + '/Dev.txt', sep=',', names=['label', 'video'])
    p1_test_df = pd.read_csv(protocols_folder[0] + '/Test.txt', sep=',', names=['label', 'video'])

    p1_output_csvs = [
        protocols_folder[0] + "/train.csv",
        protocols_folder[0] + "/dev.csv",
        protocols_folder[0] + "/test.csv"
    ]
    train_data = {
        'path': [],
        'target': []
    }

    dev_data = {
        'path': [],
        'target': []
    }

    test_data = {
        'path': [],
        'target': []
    }

    # protocol 1 - train
    pbar = tqdm(p1_train_df.itertuples())
    pbar.set_description("Processing Protocol 1 training")
    for row in pbar:
        # get label
        label = 1 if row.label == 1 else 0

        # get images in folder
        images = glob.glob(train_folder + "/" + row.video + "/*.jpg")

        # add all images to dict
        for image in images:
            train_data['path'].append(image)
            train_data['target'].append(label)

    train_df = pd.DataFrame(train_data)
    train_df.to_csv(p1_output_csvs[0], index=False)

    # protocol 1 - dev
    pbar = tqdm(p1_dev_df.itertuples())
    pbar.set_description("Processing Protocol 1 dev")
    for row in pbar:
        # get label
        label = 1 if row.label == 1 else 0

        # get images in folder
        images = glob.glob(dev_folder + "/" + row.video + "/*.jpg")

        # add all images to dict
        for image in images:
            dev_data['path'].append(image)
            dev_data['target'].append(label)

    dev_df = pd.DataFrame(dev_data)
    dev_df.to_csv(p1_output_csvs[1], index=False)

    # protocol 1 - test
    pbar = tqdm(p1_test_df.itertuples())
    pbar.set_description("Processing Protocol 1 test")
    for row in pbar:
        # get label
        label = 1 if row.label == 1 else 0

        # get images in folder
        images = glob.glob(test_folder + "/" + row.video + "/*.jpg")

        # add all images to dict
        for image in images:
            test_data['path'].append(image)
            test_data['target'].append(label)

    test_df = pd.DataFrame(test_data)
    test_df.to_csv(p1_output_csvs[2], index=False)
