from typing import Callable
import os
import numpy as np
from PIL import Image
import torch
import logging

from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def scale(img, min: float = -1.0, max: float = 1.0):
    img_std = (img - 0) / (255 - 0)
    img_scaled = img_std * (max - min) + min

    return img_scaled.astype(np.float32)


class MinMaxScale(A.ImageOnlyTransform):
    """
    Args:
        min_value (float, list of float): minimum of range
        max_value  (float, list of float): maximum of range

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self, min_value: float = -1.0, max_value: float = 1.0
    ):
        super(MinMaxScale, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, image, **params):
        return scale(image, self.min_value, self.max_value)

    def get_transform_init_args_names(self):
        return ("min", "max")


def get_train_augmentations(image_size: int = 224, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)):
    return A.Compose(
        [
            # A.RandomBrightnessContrast(brightness_limit=32, contrast_limit=(0.5, 1.5)),
            # A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=(1, 2)),
            # A.CoarseDropout(20),
            A.Rotate(10),

            A.Resize(image_size, image_size),
            # A.RandomCrop(image_size, image_size, p=0.5),

            A.LongestMaxSize(image_size),
            MinMaxScale(),
            A.HorizontalFlip(),
            A.PadIfNeeded(image_size, image_size, 0),
            # A.Transpose(),
            ToTensor(),
        ]
    )


def get_test_augmentations(image_size: int = 224, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.LongestMaxSize(image_size),
            MinMaxScale(),
            A.PadIfNeeded(image_size, image_size, 0),
            ToTensor(),
        ]
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: "pd.DataFrame",
        root: str,
        transforms: Callable,
        face_detector: dict = None,
        with_labels: bool = True,
    ):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transforms = transforms
        self.with_labels = with_labels
        self.face_extractor = None
        if face_detector is not None:
            face_detector["keep_all"] = True
            face_detector["post_process"] = False
            self.face_extractor = MTCNN(**face_detector)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int):
        # updated for absolute paths
        path = self.df.iloc[item].path

        image = Image.open(path)
        original_image = image

        if self.with_labels:
            target = self.df.iloc[item].target

        if self.face_extractor is not None:
            faces, probs = self.face_extractor(image, return_prob=True)
            if faces is None:
                logging.warning(f"{path} doesn't containt any face!")
                image = self.transforms(image=np.array(image))["image"]
                if self.with_labels:
                    return image, target
                else:
                    return image
            if faces.shape[0] != 1:
                logging.warning(
                    f"{path} - {faces.shape[0]} faces detected"
                )
                face = (
                    faces[np.argmax(probs)]
                    .numpy()
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
            else:
                face = faces[0].numpy().astype(np.uint8).transpose(1, 2, 0)
            image = self.transforms(image=face)["image"]
        else:
            image = self.transforms(image=np.array(image))["image"]

        if self.with_labels:
            return image, target
        else:
            # output = torch.cat((image, original_image), 0)
            #
            # assert torch.equal(output[:3], image)
            # assert torch.equal(output[3:], original_image)

            return image


class FASD(torch.utils.data.Dataset):
    def __init__(
        self,
        df: "pd.DataFrame",
        root: str,
        transforms: Callable,
        face_detector: dict = None,
        with_labels: bool = True,
    ):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transforms = transforms
        self.with_labels = with_labels
        self.face_extractor = None
        if face_detector is not None:
            face_detector["keep_all"] = True
            face_detector["post_process"] = False
            self.face_extractor = MTCNN(**face_detector)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int):
        # casia-fasd csvs use absolute paths so joining with the root is unnecessary
        path = self.df.iloc[item].path

        image = Image.open(path)
        if self.with_labels:
            target = self.df.iloc[item].target

        if self.face_extractor is not None:
            faces, probs = self.face_extractor(image, return_prob=True)
            if faces is None:
                logging.warning(f"{path} doesn't containt any face!")
                image = self.transforms(image=np.array(image))["image"]
                if self.with_labels:
                    return image, target
                else:
                    return image
            if faces.shape[0] != 1:
                logging.warning(
                    f"{path} - {faces.shape[0]} faces detected"
                )
                face = (
                    faces[np.argmax(probs)]
                    .numpy()
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
            else:
                face = faces[0].numpy().astype(np.uint8).transpose(1, 2, 0)
            image = self.transforms(image=face)["image"]
        else:
            image = self.transforms(image=np.array(image))["image"]

        if self.with_labels:
            return image, target
        else:
            return image


class SIWM(torch.utils.data.Dataset):
    def __init__(
        self,
        df: "pd.DataFrame",
        root: str,
        transforms: Callable,
        face_detector: dict = None,
        with_labels: bool = True,
    ):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transforms = transforms
        self.with_labels = with_labels
        self.face_extractor = None
        if face_detector is not None:
            face_detector["keep_all"] = True
            face_detector["post_process"] = False
            self.face_extractor = MTCNN(**face_detector)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int):
        # siwm csvs use absolute paths so joining with the root is unnecessary
        path = self.df.iloc[item].path

        image = Image.open(path)
        if self.with_labels:
            target = self.df.iloc[item].target

        if self.face_extractor is not None:
            faces, probs = self.face_extractor(image, return_prob=True)
            if faces is None:
                logging.warning(f"{path} doesn't containt any face!")
                image = self.transforms(image=np.array(image))["image"]
                if self.with_labels:
                    return image, target
                else:
                    return image
            if faces.shape[0] != 1:
                logging.warning(
                    f"{path} - {faces.shape[0]} faces detected"
                )
                face = (
                    faces[np.argmax(probs)]
                    .numpy()
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
            else:
                face = faces[0].numpy().astype(np.uint8).transpose(1, 2, 0)
            image = self.transforms(image=face)["image"]
        else:
            image = self.transforms(image=np.array(image))["image"]

        if self.with_labels:
            return image, target
        else:
            return image
