import argparse
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd


if __name__ == "__main__":
    device = "cuda:2" if torch.cuda.is_available() else None

    parser = argparse.ArgumentParser(
            description='Detect faces using MTCNN.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-o', '--output_size', type=int, help='Output size of cropped images.', default=224)
    parser.add_argument('-l', '--labels_file', type=str, help='The file to refer to for the labels.', default=None)

    args = parser.parse_args()

    if args.labels_file is None:
        args.labels_file = "/root/datasets/casia-fasd/labels.csv"

    labels_df = pd.read_csv(args.labels_file)
    face_detector = MTCNN(image_size=args.output_size, device=device)

    pbar = tqdm(range(len(labels_df)))
    for index in pbar:
        path = labels_df.iloc[index].path
        pbar.set_description("Processing image %s" % path)
        image = Image.open(path)

        cropped_image = face_detector(image, save_path=path)



