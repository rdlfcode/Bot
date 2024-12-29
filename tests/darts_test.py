# Imports

import os
import torch
import numpy as np
import pandas as pd
import darts_tools as dt
from darts import TimeSeries, concatenate
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Setup and Data Processing

# Load settings and data
settings = dt.load_settings()

# Create copy in .py file format
dt.ipynb_to_py("darts_test")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_CUDA_ALLOC_SYNC'] = "1"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision(settings["float_precision"])
torch.set_grad_enabled(True)

df = pd.read_csv(settings["data"])
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Define columns and create TimeSeries objects for each ticker
value_cols = [col for col in df.columns if col not in settings["protected_columns"]]

stocks = TimeSeries.from_group_dataframe(
    df,
    group_cols="Ticker",
    time_col="Datetime",
    value_cols=value_cols,
)

# Preparing the data
scalers, trains, vals, past_covariates, future_covariates = [], [], [], [], []

# Assuming past covariates are all columns except for 'target'
past_covariate_cols = [col for col in value_cols if col != settings["target"]]

for stock in stocks:
    split = int(len(stock) * settings["train_test_pct"])
    train, val = stock[:split], stock[split:]
    
    # Transforming both past covariates and target series
    scaler = Scaler()
    target_train_transformed = scaler.fit_transform(train[settings["target"]])
    target_val_transformed = scaler.transform(val[settings["target"]])
    
    # Generate future covariates
    future_cov = datetime_attribute_timeseries(stock.time_index, attribute='week', one_hot=False, add_length=settings["max_prediction_length"])
    future_cov = future_cov.stack(datetime_attribute_timeseries(stock.time_index, attribute='month', one_hot=False, add_length=settings["max_prediction_length"]))
    future_cov = future_cov.stack(datetime_attribute_timeseries(stock.time_index, attribute='quarter', one_hot=False, add_length=settings["max_prediction_length"]))
    future_cov = future_cov.astype(np.float32)

    cov_scaler = Scaler()
    train_future_cov, val_future_cov = future_cov[:len(train)], future_cov[len(train):]
    tf_train_future_cov = cov_scaler.fit_transform(train_future_cov)
    tf_val_future_cov = cov_scaler.transform(val_future_cov)
    future_cov = concatenate([tf_train_future_cov, tf_val_future_cov])

    # Generate past covariates
    train_past_cov = cov_scaler.fit_transform(train[past_covariate_cols])
    val_past_cov = cov_scaler.transform(val[past_covariate_cols])
    past_cov = concatenate([train_past_cov, val_past_cov])

    scalers.append(scaler)
    trains.append(target_train_transformed)
    vals.append(target_val_transformed)
    past_covariates.append(past_cov)
    future_covariates.append(future_cov)

# Configure Network, Trainer, and Model

import tensorflow as tf
import tensorboard as tb

# Setup loggers
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

# Define the TFTModel with specified settings
model = TFTModel(
    input_chunk_length=len(train) // settings["n_splits"],
    output_chunk_length=settings["max_prediction_length"],
    hidden_size=settings["hidden_size"],
    lstm_layers=settings["lstm_layers"],
    num_attention_heads=settings["attention_head_size"],
    dropout=settings["dropout"],
    batch_size=settings["batch_size"],
    n_epochs=settings["max_train_epochs"],
    likelihood=QuantileRegression(),
    optimizer_kwargs={'lr': settings["learning_rate"]},
    feed_forward=settings["feed_forward"],
    loss_fn=settings["loss_func"],
    add_relative_index=True,
    add_encoders={
        'cyclic': {'future': ['week', 'month', 'quarter'], 'past': ['week', 'month', 'quarter']},
        'datetime_attribute': {'future': ['week', 'month', 'quarter'], 'past': ['week', 'month', 'quarter']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'custom': {'future': [dt.encode_us_holidays], 'past': [dt.encode_us_holidays]},
        'transformer': Scaler()
    },
    pl_trainer_kwargs={"accelerator":           settings["accelerator"], 
                       "devices":               -1,
                       "gradient_clip_val":     settings["gradient_clip_val"],
                       "callbacks":             [lr_logger,
                                                 early_stop_callback,
                                                 ckpt_callback,],
                        "logger":               logger,
                        "enable_checkpointing": True,
                       },
    random_state=42,
    log_tensorboard=True,
    work_dir=settings["base_path"],
    use_static_covariates=False,
)

# Fit the model
model.fit(
    series=trains,
    past_covariates=past_covariates,
    future_covariates=future_covariates,
    val_series=vals,
    val_past_covariates=past_covariates,
    val_future_covariates=future_covariates,
    verbose=True,
)

# Predict and evaluate
evaluations = []
for train_series, val_series in zip(trains, vals):
    pred_series = model.predict(n=settings["max_prediction_length"],
                                series=train_series,
                                num_loader_workers=20,
                                num_jobs=-1,
                                )
    evaluations.append((mape(val_series, pred_series), rmse(val_series, pred_series)))

# Calculating average MAPE and RMSE for simplicity
average_mape = sum([eval[0] for eval in evaluations]) / len(evaluations)
average_rmse = sum([eval[1] for eval in evaluations]) / len(evaluations)

print(f"Average MAPE: {average_mape}")
print(f"Average RMSE: {average_rmse}")

# Save the model
model.save(settings["model_path"])

