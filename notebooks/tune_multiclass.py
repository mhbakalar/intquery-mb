import pandas as pd
import numpy as np
import lightning.pytorch as pl

from ray.train.torch import TorchTrainer

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from lit_modules import data_modules
from lit_modules import modules

if __name__ == "__main__":
    data_path = '~/code/cryptic/cryptic/data/TB000208a'
    genomic_reference_file = '~/code/cryptic/cryptic/data/references/hg38.fa'
    n_classes = 2
    seq_length = 46
    vocab_size = 4
    input_size = seq_length*vocab_size
    hidden_size = 512
    n_hidden = 2
    train_test_split = 0.8

    # Model tuning
    def train_func(config):

        # Build the lightning modules
        data_module = data_modules.MulticlassDataModule(data_path, threshold=config['threshold'], n_classes=n_classes, train_test_split=train_test_split, batch_size=64)
        lit_model = modules.Classifier(input_size=input_size, hidden_size=config['hidden_size'], n_classes=n_classes, n_hidden=n_hidden, dropout=0.5, lr=config['lr'])

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(lit_model, datamodule=data_module)

    search_space = {
        "hidden_size": tune.choice([256, 512, 1024, 2048]),
        "lr": tune.choice([5e-3, 1e-3]),
        "threshold": tune.loguniform(1e-4, 1e-1)
    }

    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = 50

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_acc_step",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_mnist_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="val_acc_step",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()


    results = tune_mnist_asha(num_samples=num_samples)
