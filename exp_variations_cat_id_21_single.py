import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import seaborn as sns
import matplotlib.pyplot as plt

import logging
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

device = "cuda" if torch.cuda.is_available() else "cpu"

categorical_id = 21

time_df = pd.read_csv('../../../data/processed_electricity_data.csv')
time_df = time_df[time_df['id'].isin([f"MT_{str(i).zfill(3)}" for i in range(1, categorical_id)])]


valid_boundary=1315
train_data = time_df[time_df["days_from_start"] <= valid_boundary].copy()
val_data = time_df[time_df["days_from_start"] > valid_boundary].copy()
time_df = pd.concat([train_data, val_data]).sort_values(by=["categorical_id", "hours_from_start"]).reset_index(drop=True)

max_prediction_length = 24
max_encoder_length = 7*24
training_cutoff = time_df["hours_from_start"].max() - max_prediction_length

training = TimeSeriesDataSet(
    time_df[lambda x: x.days_from_start <= valid_boundary],
    time_idx="hours_from_start",
    target="power_usage",
    group_ids=["categorical_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["categorical_id"],
    time_varying_known_reals=["hours_from_start","day","day_of_week", "month", 'hour'],
    time_varying_unknown_reals=['power_usage'],
    target_normalizer=GroupNormalizer(
        groups=["categorical_id"], transformation=None
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


validation = TimeSeriesDataSet.from_dataset(training, time_df, predict=True, stop_randomization=True)

# create dataloaders for  our model
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)


baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
print(MAE()(baseline_predictions.output, baseline_predictions.y))




early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs", name="new_experiment2_short/", default_hp_metric=False)  # Log results to TensorBoard


trainer = pl.Trainer(
    max_epochs=30,
    accelerator='gpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=3.0,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=1,
    dropout=1,
    hidden_continuous_size=16,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=1,
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt

# Specify the TensorBoard log directory
log_dir = "/home/ubuntu/Electricity Dataset/FE_Experiments/exp2_short/lightning_logs/new_experiment2_short/version_5"  # Replace 'X' with the correct version
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Extract training and validation losses
train_loss_data = event_acc.Scalars("train_loss_epoch")
val_loss_data = event_acc.Scalars("val_loss")

# Convert to DataFrame for easier handling
train_loss_df = pd.DataFrame({
    "step": [x.step for x in train_loss_data],
    "value": [x.value for x in train_loss_data],
    "wall_time": [x.wall_time for x in train_loss_data]
})

val_loss_df = pd.DataFrame({
    "step": [x.step for x in val_loss_data],
    "value": [x.value for x in val_loss_data],
    "wall_time": [x.wall_time for x in val_loss_data]
})

print("Training Loss Data:\n", train_loss_df.head())
print("Validation Loss Data:\n", val_loss_df.head())

import matplotlib.pyplot as plt

# Normalize steps to epochs
train_loss_df["epoch"] = range(1, len(train_loss_df) + 1)
val_loss_df["epoch"] = range(1, len(val_loss_df) + 1)

# Plot the loss curves
plt.figure(figsize=(10, 6))

# Plot training loss
plt.plot(train_loss_df["epoch"], train_loss_df["value"], label="Training Loss", color="blue", marker='o')

# Plot validation loss
plt.plot(val_loss_df["epoch"], val_loss_df["value"], label="Validation Loss", color="orange", marker='o')

# Add labels, title, and legend
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Epochs")
plt.legend()
plt.grid()

# Show the plot
plt.show()

