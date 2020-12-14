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
        description='Build protocol splits for 3 and 4.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('root_folder', type=str, help='Absolute path to root folder of dataset.')
    parser.add_argument('-p', '--protocol', type=int, required=True, help='Protocol to build.')
    parser.add_argument('-d', '--debug', type=bool, help='Debug mode.', default=False)
    args = parser.parse_args()

    if args.protocol < 3 or args.protocol > 4:
        raise Exception("Only protocols 3 and 4 are allowed for this file.")

    # load csvs
    train_csv = "/root/datasets/oulu/train.csv"
    dev_csv = "/root/datasets/oulu/dev.csv"
    test_csv = "/root/datasets/oulu/test.csv"

    train_folder = args.root_folder.rstrip("/") + "/Train_files/imgs"
    dev_folder = args.root_folder.rstrip("/") + "/Dev_files/imgs"
    test_folder = args.root_folder.rstrip("/") + "/Test_files/imgs"

    protocol_root_folder = args.root_folder.rstrip("/") + "/Protocols"
    protocol = int(args.protocol)
    protocol_folder = protocol_root_folder + "/Protocol_" + str(protocol)

    make_folder(protocol_root_folder)
    make_folder(protocol_folder)

    train_dfs = []
    dev_dfs = []
    test_dfs = []
    
    for i in range(1, 7):
        train_df = pd.read_csv(protocol_folder + '/Train_' + str(i) + '.txt', sep=',', names=['label', 'video'])
        dev_df = pd.read_csv(protocol_folder + '/Dev_' + str(i) + '.txt', sep=',', names=['label', 'video'])
        test_df = pd.read_csv(protocol_folder + '/Test_' + str(i) + '.txt', sep=',', names=['label', 'video'])

        output_csvs = [
            protocol_folder + "/train_" + str(i) + ".csv",
            protocol_folder + "/dev_" + str(i) + ".csv",
            protocol_folder + "/test_" + str(i) + ".csv"
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
        pbar = tqdm(train_df.itertuples())
        pbar.set_description("Processing Protocol %d training for user %d" % (protocol, i))
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
        train_df.to_csv(output_csvs[0], index=False)

        # protocol 1 - dev
        pbar = tqdm(dev_df.itertuples())
        pbar.set_description("Processing Protocol %d dev" % protocol)
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
        dev_df.to_csv(output_csvs[1], index=False)

        # protocol 1 - test
        pbar = tqdm(test_df.itertuples())
        pbar.set_description("Processing Protocol %d test" % protocol)
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
        test_df.to_csv(output_csvs[2], index=False)

        print("")

        # train_dfs.append(train_df)
        # dev_dfs.append(dev_df)
        # test_dfs.append(test_df)
