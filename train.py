from argparse import ArgumentParser, Namespace

import safitty
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt

from pl_model import LightningModel


# Seems we have to smake a plot as the first thing for plots to work at all?

plt.plot([1, 2, 3])
plt.plot([3, 2, 1])
plt.show()

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    configs = Namespace(**configs)

    if not os.path.isdir(configs.default_root_dir):
        os.makedirs(configs.default_root_dir)

    model = LightningModel(hparams=configs)
    trainer = pl.Trainer.from_argparse_args(
        configs,
        fast_dev_run=False,
        early_stop_callback=True,
        default_root_dir=configs.default_root_dir,
        gpus=1,
        # distributed_backend="ddp"
    )
    trainer.fit(model)
