import glob
import pandas as pd
import argparse
from tqdm import tqdm
import os

"""
Expected folder structure:
    root_folder
    |---train
    |---|----live
    |---|----spoof
    |---val
    ...
"""

if __name__ == "__main__":
    print("Starting...")
    default_output_path = '/root/datasets/siwm/'

    if not os.path.isdir(default_output_path):
        os.makedirs(default_output_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder_path", required=True, help="Absolute path to folder in which to search.")
    parser.add_argument("-o", "--output_folder", required=False, default=default_output_path,
                        help="Absolute path to folder containing csv label output.")
    args = parser.parse_args()

    args.folder_path = args.folder_path.rstrip("/")

    print("Processing train folder...")
    train_live_path = args.folder_path + "/train/live"
    train_live_folder_list = glob.glob(train_live_path + '/*')
    train_spoof_path = args.folder_path + "/train/spoof"
    train_spoof_folder_list = glob.glob(train_spoof_path + '/*')

    print("Processing val folder...")
    val_live_path = args.folder_path + "/val/live"
    val_live_folder_list = glob.glob(val_live_path + '/*')
    val_spoof_path = args.folder_path + "/val/spoof"
    val_spoof_folder_list = glob.glob(val_spoof_path + '/*')

    print("Processing test folder...")
    test_live_path = args.folder_path + "/test/live"
    test_live_folder_list = glob.glob(test_live_path + '/*')
    test_spoof_path = args.folder_path + "/test/spoof"
    test_spoof_folder_list = glob.glob(test_spoof_path + '/*')
    
    print("Detected train live images: ", len(train_live_folder_list))
    print("Detected train spoof images: ", len(train_spoof_folder_list))
    print("Detected val live images: ", len(val_live_folder_list))
    print("Detected val spoof images: ", len(val_spoof_folder_list))
    print("Detected test live images: ", len(test_live_folder_list))
    print("Detected test spoof images: ", len(test_spoof_folder_list))
    
    data = {
        'path': [],
        'target': []
    }

    # Train Live
    pbar = tqdm(train_live_folder_list, desc="Train Live")
    for folder in pbar:
        image_path = glob.glob(folder + '/*.png')[0]
        data['path'].append(image_path)
        data['target'].append(0)

    # Train Spoof
    pbar = tqdm(train_spoof_folder_list, desc="Train Spoof")
    for folder in pbar:
        image_path = glob.glob(folder + '/*.png')[0]
        data['path'].append(image_path)
        data['target'].append(1)

    train_df = pd.DataFrame(data)
    train_df.to_csv(args.output_folder + "train_labels.csv", index=False)

    print("")

    val_data = {
        'path': [],
        'target': []
    }

    # Val Live
    pbar = tqdm(val_live_folder_list, desc="Val Live")
    for folder in pbar:
        image_path = glob.glob(folder + '/*.png')[0]
        val_data['path'].append(image_path)
        val_data['target'].append(0)

    # Val spoof
    pbar = tqdm(val_spoof_folder_list, desc="Val Spoof")
    for folder in pbar:
        image_path = glob.glob(folder + '/*.png')[0]
        val_data['path'].append(image_path)
        val_data['target'].append(1)

    val_df = pd.DataFrame(val_data)
    val_df.to_csv(args.output_folder + "val_labels.csv", index=False)

    print("")

    test_data = {
        'path': [],
        'target': []
    }

    # Test Live
    pbar = tqdm(test_live_folder_list, desc="Test Live")
    for folder in pbar:
        image_path = glob.glob(folder + '/*.png')[0]
        test_data['path'].append(image_path)
        test_data['target'].append(0)

    # Test Spoof
    pbar = tqdm(test_spoof_folder_list, "Test Spoof")
    for folder in pbar:
        image_path = glob.glob(folder + '/*.png')[0]
        test_data['path'].append(image_path)
        test_data['target'].append(1)

    test_df = pd.DataFrame(test_data)
    test_df.to_csv(args.output_folder + "test_labels.csv", index=False)




