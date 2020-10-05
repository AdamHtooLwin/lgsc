import pandas as pd
import numpy as np

from sklearn import metrics

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from catalyst.data.sampler import BalanceClassSampler
from catalyst.contrib.nn.criterion.focal import FocalLossMultiClass
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from datasets import Dataset, get_test_augmentations, get_train_augmentations
from models.scan import SCAN
from loss import TripletLoss
from metrics import eval_from_scores
from utils import GridMaker


def imshow(batch, title=None):
    """Imshow for Tensor."""
    images = torchvision.utils.make_grid(batch)
    images = images.detach().cpu().numpy().transpose((1, 2, 0))
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def construct_grid(batch):
    images = torchvision.utils.make_grid(batch)
    images = images.detach().cpu().numpy()
    return images


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = SCAN()
        self.triplet_loss = TripletLoss()
        self.log_cues = not self.hparams.cue_log_every == 0
        self.grid_maker = GridMaker()
        if self.hparams.use_focal_loss:
            self.clf_criterion = FocalLossMultiClass()
        else:
            self.clf_criterion = nn.CrossEntropyLoss()

    def get_progress_bar_dicts(self):
        items = super().get_progress_bar_dicts()
        return items

    def forward(self, x):
        return self.model(x)

    def infer(self, x):
        outs, _ = self.model(x)
        return outs[-1]

    def calc_losses(self, outs, clf_out, target):

        clf_loss = (
            self.clf_criterion(clf_out, target)
            * self.hparams.loss_coef["clf_loss"]
        )
        cue = outs[-1]
        cue = target.reshape(-1, 1, 1, 1) * cue
        num_reg = (
            torch.sum(target) * cue.shape[1] * cue.shape[2] * cue.shape[3]
        ).type(torch.float)
        reg_loss = (
            torch.sum(torch.abs(cue)) / (num_reg + 1e-9)
        ) * self.hparams.loss_coef["reg_loss"]

        trip_loss = 0
        bs = outs[-1].shape[0]
        for feat in outs[:-1]:
            feat = F.adaptive_avg_pool2d(feat, [1, 1]).view(bs, -1)
            trip_loss += (
                self.triplet_loss(feat, target)
                * self.hparams.loss_coef["trip_loss"]
            )
        total_loss = clf_loss + reg_loss + trip_loss

        return total_loss, clf_loss, reg_loss, trip_loss

    def training_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]

        outs, clf_out = self(input_)
        loss, clf_loss, reg_loss, trip_loss = self.calc_losses(outs, clf_out, target)

        if self.hparams.show_imgs:
            imshow(input_, title="Training")
            imshow(outs[-1], title="Training Cues")

        if self.log_cues:
            if self.current_epoch % self.hparams.cue_log_every == 0:
                images_grid = construct_grid(input_)
                cues_grid = construct_grid(outs[-1])

                self.logger.experiment.add_image(
                    "training_cues", cues_grid, self.current_epoch * batch_idx
                )
                self.logger.experiment.add_image(
                    "training_images", images_grid, self.current_epoch * batch_idx
                )
        # if self.log_cues:
        #     if (self.current_epoch * batch_idx) % self.hparams.cue_log_every == 0:
        #         images_grid = construct_grid(input_)
        #         cues_grid = construct_grid(outs[-1])
        #
        #         self.logger.experiment.add_image(
        #             "training_cues", cues_grid, self.current_epoch * batch_idx
        #         )
        #         self.logger.experiment.add_image(
        #             "training_images", images_grid, self.current_epoch * batch_idx
        #         )

        tensorboard_logs = {
            "train_loss": loss,
            "clf_loss": clf_loss,
            "reg_loss": reg_loss,
            "trip_loss": trip_loss
        }
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "train_avg_loss": avg_loss,
        }
        return {"train_avg_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]

        if self.hparams.show_imgs:
            imshow(batch[0], title="Validation")

        outs, clf_out = self(input_)
        loss, clf_loss, reg_loss, trip_loss = self.calc_losses(outs, clf_out, target)
        # result = pl.EvalResult()
        # result.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # result.log('score', clf_out.cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # result.log('target', target.cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        tensorboard_logs = {
            "val_loss": loss,
            "val_clf_loss": clf_loss,
            "val_reg_loss": reg_loss,
            "val_trip_loss": trip_loss,
        }

        val_dict = {
            "val_loss": loss,
            "val_clf_loss": clf_loss,
            "val_reg_loss": reg_loss,
            "val_trip_loss": trip_loss,
            "score": clf_out.cpu().numpy(),
            "target": target.cpu().numpy(),
            "log": tensorboard_logs
        }

        if self.log_cues:
            if self.current_epoch % self.hparams.cue_log_every == 0:
                images_grid = construct_grid(input_)
                cues_grid = construct_grid(outs[-1])

                self.logger.experiment.add_image(
                    "val_cues", cues_grid, self.current_epoch * batch_idx
                )
                self.logger.experiment.add_image(
                    "val_images", images_grid, self.current_epoch * batch_idx
                )

        return val_dict

    # outputs of all the training steps -> list of dicts
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        targets = np.hstack([output["target"] for output in outputs])
        scores = np.vstack([output["score"] for output in outputs])[:, 1]
        metrics_, best_thr, acc = eval_from_scores(scores, targets)
        acer, apcer, npcer = metrics_
        roc_auc = metrics.roc_auc_score(targets, scores)
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_roc_auc": roc_auc,
            "val_acer": acer,
            "val_apcer": apcer,
            "val_npcer": npcer,
            "val_acc": acc,
            "val_thr": best_thr,
        }
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=self.hparams.milestones, gamma=self.hparams.gamma
        )
        return [optim], [scheduler]

    def train_dataloader(self):
        mean = (self.hparams.mean['r'], self.hparams.mean['g'], self.hparams.mean['b'])
        std = (self.hparams.std['r'], self.hparams.std['g'], self.hparams.std['b'])

        transforms = get_train_augmentations(self.hparams.image_size, mean=mean, std=std)
        df = pd.read_csv(self.hparams.train_df)
        try:
            face_detector = self.hparams.face_detector
        except AttributeError:
            face_detector = None
        dataset = Dataset(
            df, self.hparams.path_root, transforms, face_detector=face_detector
        )
        if self.hparams.use_balance_sampler:
            labels = list(df.target.values)
            sampler = BalanceClassSampler(labels, mode="upsampling")
            shuffle = False
        else:
            sampler = None
            shuffle = True
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_train,
            sampler=sampler,
            shuffle=shuffle,
        )
        return dataloader

    def val_dataloader(self):
        mean = (self.hparams.mean['r'], self.hparams.mean['g'], self.hparams.mean['b'])
        std = (self.hparams.std['r'], self.hparams.std['g'], self.hparams.std['b'])

        transforms = get_test_augmentations(self.hparams.image_size, mean=mean, std=std)
        df = pd.read_csv(self.hparams.val_df)
        try:
            face_detector = self.hparams.face_detector
        except AttributeError:
            face_detector = None
        dataset = Dataset(
            df, self.hparams.path_root, transforms, face_detector=face_detector
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_val,
            shuffle=False,
        )
        return dataloader
