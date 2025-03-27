import os
import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


categorical_id = 50

time_df = pd.read_csv('../../../data/processed_electricity_data.csv')
time_df = time_df[time_df['id'].isin([f"MT_{str(i).zfill(3)}" for i in range(1, categorical_id)])]


valid_boundary=1315
train_data = time_df[time_df["days_from_start"] <= valid_boundary].copy()
val_data = time_df[time_df["days_from_start"] > valid_boundary].copy()
time_df = pd.concat([train_data, val_data]).sort_values(by=["categorical_id", "hours_from_start"]).reset_index(drop=True)
time_df['hours_from_start'] = time_df['hours_from_start'].astype(int)

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

# Define hyperparameter variations
learning_rates = [0.001, 0.005, 0.01]
dropouts = [0.1, 0.5, 1.0]
gradient_clip_vals = [0.5, 1.0, 5.0]
hidden_sizes = [8, 16]
hidden_continuous_sizes = [8, 16]
attention_head_sizes = [2, 3]

# Create folders for models and plots
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Generate all hyperparameter combinations
combinations = list(itertools.product(learning_rates, dropouts, gradient_clip_vals, 
                                      hidden_sizes, hidden_continuous_sizes, attention_head_sizes))

# Training loop
for i, (lr, dropout, grad_clip, hidden_size, hidden_cont_size, attn_heads) in enumerate(combinations):
    experiment_config = f"lr={lr}, dropout={dropout}, grad_clip={grad_clip}, hidden_size={hidden_size}, hidden_cont_size={hidden_cont_size}, attn_heads={attn_heads}"
    print("===================================================================")
    print(f"Training model {i+1}/{len(combinations)} with {experiment_config}")
    print("===================================================================")
    
    logger = TensorBoardLogger("lightning_logs", name=f"experiment_{i}", default_hp_metric=False)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_logger = LearningRateMonitor()
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=grad_clip,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger
    )
    
    # Define model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attn_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_cont_size,
        output_size=7,  # default quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=1,
    )
    
    # Train model
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    
    # Extract training and validation loss from TensorBoard logs
    log_dir = f"lightning_logs/experiment_{i}/version_0"
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    train_loss_data = event_acc.Scalars("train_loss_epoch")
    val_loss_data = event_acc.Scalars("val_loss")
    
    train_loss_df = pd.DataFrame({
        "epoch": range(1, len(train_loss_data) + 1),
        "train_loss": [x.value for x in train_loss_data]
    })
    
    val_loss_df = pd.DataFrame({
        "epoch": range(1, len(val_loss_data) + 1),
        "val_loss": [x.value for x in val_loss_data]
    })
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_df["epoch"], train_loss_df["train_loss"], label="Training Loss", color="blue", marker='o')
    plt.plot(val_loss_df["epoch"], val_loss_df["val_loss"], label="Validation Loss", color="orange", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss (Experiment {i})")
    plt.suptitle(f"Experiment Config: {experiment_config}", fontsize=10)
    plt.legend()
    plt.grid()
    
    plot_path = f"plots/loss_plot_{i}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved at {plot_path}")
    
    # Save model
    model_path = f"models/model_{i}.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved at {model_path}")

print("All experiments completed!")
