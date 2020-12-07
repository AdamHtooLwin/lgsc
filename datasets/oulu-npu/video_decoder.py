import os
import glob
from tqdm import tqdm

import cv2
import argparse
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image


def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def delete():
    """
    Function to delete the first frame in every video
    Should not be needed in the latest frame extractor.
    :return:
    """
    labels_file = "/root/datasets/casia-fasd/train_labels.csv"

    data_df = pd.read_csv(labels_file)
    print("Initial length: ", len(data_df))

    pbar = tqdm(range(len(data_df)))
    for index in pbar:
        path = data_df.loc[index].path
        pbar.set_description("Processing index %s" % index)

        video = path.split("/")[-2]
        file = path.split("/")[-1]

        if "HR" in video and file == "1.jpg":
            data_df = data_df.drop(index)

            if os.path.exists(path):
                os.remove(path)

    output_path = "/root/datasets/casia-fasd/train_labels_filtered.csv"
    print("Final length: ", len(data_df))
    data_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    default_labels_file = "/root/datasets/oulu/labels.txt"
    default_output_csv = "/root/datasets/oulu/labels.csv"
    default_device = 'cuda:0'

    parser = argparse.ArgumentParser(
            description='Extract frames from OULU videos.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('videos_folder', type=str, help='Absolute path to folder to process videos.')
    parser.add_argument('-t', '--file_type', type=str, help='File type of video.', default="avi")
    parser.add_argument('-f', '--frames', type=int, help='Number of frames to extract.', default=5)
    parser.add_argument('-o', '--output_csv', type=str, help='File to output CSV label file.', default=default_output_csv)
    parser.add_argument('-l', '--labels_file', type=str, help='The file to refer to for the labels.', default=default_labels_file)
    parser.add_argument('-d', '--device', type=str, help='The device to run MTCNN face cropper.', default=default_device)
    parser.add_argument('-s', '--output_size', type=int, help='Output size of cropped images.', default=224)
    parser.add_argument('-de', '--debug', type=bool, help='Debug mode.', default=False)

    args = parser.parse_args()

    video_files = glob.glob(args.videos_folder + "" + "/*." + args.file_type)

    img_folder = args.videos_folder.rstrip("/") + "/imgs/"
    make_folder(img_folder)

    labels_df = pd.read_csv(args.labels_file, sep=" ")
    data = {
        'path': [],
        'target': []
    }

    face_detector = MTCNN(image_size=args.output_size, device=args.device)

    pbar = tqdm(video_files)
    for video_file in pbar:
        video_folder_name = video_file.split("/")[-1].split(".")[0]
        video_type = video_folder_name.split("_")[-1]   # get last number for video type

        label = labels_df.loc[labels_df['type'] == int(video_type)].iloc[0].target
        pbar.set_description("Processing video %s " % video_folder_name)
        video_folder = img_folder + video_folder_name
        uncropped_folder = video_folder + "/uncropped"

        make_folder(video_folder)
        make_folder(uncropped_folder)

        vidcap = cv2.VideoCapture(video_file)

        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        increment = int(length / args.frames)

        success, image = vidcap.read()
        count = 1
        while success:
            if count == 3 or count % increment == 0:
                file_name = video_folder + "/%d.jpg" % count
                uncropped_path = uncropped_folder + "/%d.jpg" % count

                data['path'].append(file_name)
                data['target'].append(label)

                cv2.imwrite(uncropped_path, image)  # save uncropped frame

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image)
                cropped_image = face_detector(image_pil, save_path=file_name)   # save cropped frame

            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        vidcap.release()

        if args.debug and video_type == '5':
            break

    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)


