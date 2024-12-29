import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import torch
import pandas as pd
import model_tools as mt
from settings import settings

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import QuantileLoss, MAPE, RMSE, MAE, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

settings = settings["study"]
settings["protected_columns"] = ["Datetime", "Ticker", settings["target"]]
loss_funcs = {
    "val_SMAPE": SMAPE(), 
    "val_MAE":   MAE(),
    "val_RMSE":  RMSE(),
    "val_MAPE":  MAPE(),
}
# Dynamic path updates
settings["loss_func"] = loss_funcs.get(settings["loss_name"], QuantileLoss())

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_CUDA_ALLOC_SYNC'] = "1"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision(settings["float_precision"]) # use 'high' for more accurate models
torch.set_grad_enabled(True)
print(settings["accelerator"])

# Create copy in .py file format
mt.ipynb_to_py(f"{settings['file']}.ipynb", f"{settings['file']}.py")

# Load and preprocess data
original = pd.read_csv(settings["data"])

# Preprocess data, settings and check correlations
data, settings = mt.preprocess(original, settings) 

# Create dataset
max_encoder_length = len(data) // 4 # should be set to capture pattern in the temporal data
training_cutoff = data["Datetime"].max() - settings["max_prediction_length"]
unknown_reals = list(data.columns.drop(['Datetime', 'Ticker']))

training = TimeSeriesDataSet(
    data[lambda x: x.Datetime <= training_cutoff],
    time_idx="Datetime",
    target=settings["target"], 
    group_ids=["Ticker"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=settings["max_prediction_length"],
    static_reals=["Ticker"],
    time_varying_unknown_reals=unknown_reals,
    add_relative_time_idx=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# Create dataloaders
train_dataloader = training.to_dataloader(
    train=True, 
    batch_size=settings["batch_size"], 
    num_workers=20
)
val_dataloader = validation.to_dataloader(
    train=False,
    batch_size=settings["batch_size"] * 10,
    num_workers=20
)

# Find and Load Previous Study

# Check if the file exists
last_study = None
if os.path.exists(settings['study_path']):
    with open(settings['study_path'], "rb") as f:
        last_study = pickle.load(f)
    print("Study loaded successfully.")

# Start Study

# configure network and trainer
early_stop_callback = EarlyStopping(
    monitor=settings["loss_name"],
    patience=10,
)
ckpt_callback = ModelCheckpoint(
    monitor=settings["loss_name"],
    dirpath=f"{settings['base_path']}/{settings['name']}/study_ckpts/",
    filename='{val_loss:.3f}-{epoch:02d}',
    every_n_epochs=1,
)
logger = TensorBoardLogger(
    save_dir=f"lightning_logs/lstm_layers_{settings['lstm_layers']}/", 
    name=settings["name"],
)
lr_logger = LearningRateMonitor()

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    name=settings["name"],
    lstm_layers=settings["lstm_layers"],
    timeout=settings["timeout"],
    study=last_study,
    n_trials=settings["n_trials"],
    max_epochs=settings["max_epochs"],
    gradient_clip_val_range=settings["gradient_clip_val_range"],
    hidden_size_range=settings["hidden_size_range"], 
    hidden_continuous_size_range=settings["hidden_continuous_size_range"],
    attention_head_size_range=settings["attention_head_size_range"],
    learning_rate_range=settings["learning_rate_range"],
    dropout_range=settings["dropout_range"],
    loss=settings["loss_func"],   
    loss_name=settings["loss_name"],
    trainer_kwargs=dict(
        devices=-1,
        accelerator=settings["accelerator"],
        callbacks=[early_stop_callback, ckpt_callback, lr_logger],
        log_every_n_steps=settings["log_every_n_steps"],
        limit_train_batches=settings["limit_train_batches"],
        logger=logger,
        ),
    reduce_on_plateau_patience=settings["reduce_on_plateau_patience"],
    use_learning_rate_finder=False,
    corr_thresh=settings["corr_thresh"],
    pct=settings["pct"],
    float_precision=settings["float_precision"],
    batch_size=settings["batch_size"],
    target=settings["target"],
    max_prediction_length=settings["max_prediction_length"],
    train_test_pct=settings["train_test_pct"],
    n_splits=settings["n_splits"],
    scaling_method=settings["scaling_method"],
)

# Save the trials
mt.save_study(study, settings["study_path"])

# show best hyperparameters
best_trial = study.best_trial
print(best_trial.settings)

previous_best_checkpoint = ckpt_callback.best_model_path

# configure network with new trainer and callbacks
early_stop_callback = EarlyStopping(
    monitor=settings["loss_name"],
    patience=10,
)
ckpt_callback = ModelCheckpoint(
    monitor=settings["loss_name"],
    dirpath=f"{settings['base_path']}/{settings['name']}/model_ckpts/",
    filename='{val_loss:.3f}-{epoch:02d}',
    every_n_epochs=1,
)
logger = TensorBoardLogger(
    save_dir=f"lightning_logs/lstm_layers_{settings['lstm_layers']}/", 
    name=settings["name"],
)
lr_logger = LearningRateMonitor()

print("Using model path ", previous_best_checkpoint)

best_tft = TemporalFusionTransformer.load_from_checkpoint(previous_best_checkpoint)

# Optionally, continue training for more epochs
trainer = pl.Trainer(
    max_epochs=settings["max_epochs"] * 2,
    accelerator=settings["accelerator"],
    callbacks=[early_stop_callback, ckpt_callback, lr_logger],
    logger=logger,
)
trainer.fit(best_tft, train_dataloader, val_dataloader)

# Save the model

# load the best model according to loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Save the best model
mt.save_model(best_tft, trainer, settings)

# Evaluate Performance (vs Baseline)

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
baseline_predictions = Baseline().predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator=settings["accelerator"]))
best_predictions = best_tft.predict(val_dataloader, mode="raw", return_y=True, return_x=True, trainer_kwargs=dict(accelerator=settings["accelerator"]))

# Taking the mean of the tensor
mean_baseline_mape = torch.mean(MAPE('mean').loss(baseline_predictions.output, baseline_predictions.y[0]))
mean_baseline_rmse = torch.mean(RMSE('mean').loss(baseline_predictions.output, baseline_predictions.y[0]))

# Taking the last value of the tensor
last_baseline_mape = MAPE('mean').loss(baseline_predictions.output, baseline_predictions.y[0])[:,-1]
last_baseline_rmse = RMSE('mean').loss(baseline_predictions.output, baseline_predictions.y[0])[:,-1]

# Taking the mean of the tensor
mean_tft_mape = torch.mean(MAPE('mean').loss(best_predictions.output.prediction, best_predictions.y[0]))
mean_tft_rmse = torch.mean(RMSE('mean').loss(best_predictions.output.prediction, best_predictions.y[0]))

# Taking the last value of the tensor
last_tft_mape = MAPE('mean').loss(best_predictions.output.prediction, best_predictions.y[0])[:,-1]
last_tft_rmse = RMSE('mean').loss(best_predictions.output.prediction, best_predictions.y[0])[:,-1]

print(f"Mean TFT MAPE:      {mean_tft_mape.item():.4f}")
print(f"Mean Baseline MAPE: {mean_baseline_mape.item():.4f}")
print(f"Mean TFT RMSE:      {mean_tft_rmse.item():.4f}")
print(f"Mean Baseline RMSE: {mean_baseline_rmse.item():.4f}")
print(f"Last TFT MAPE:      {last_tft_mape.item():.4f}")
print(f"Last Baseline MAPE: {last_baseline_mape.item():.4f}")
print(f"Last TFT RMSE:      {last_tft_rmse.item():.4f}")
print(f"Last Baseline RMSE: {last_baseline_rmse.item():.4f}")

best_tft.plot_prediction(best_predictions.x, best_predictions.output, idx=0, add_loss_to_title=True)
interpretation = best_tft.interpret_output(best_predictions.output, reduction="sum")
best_tft.plot_interpretation(interpretation)

if mean_tft_mape < settings["loss"]:
    import json
    settings["study"].update(best_tft.hparams)
    with open("best_settings.json", "w") as f:
        json.dump(settings, f, indent=4, default=mt.serialize)

torch.cuda.empty_cache()