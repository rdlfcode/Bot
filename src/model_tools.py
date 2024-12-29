import os
import re
import json
import torch
import pickle
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from nbformat import read
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import MAPE, RMSE, SMAPE, MAE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

def optimize_data_types(data, force_int=False):
    """
    Determines the optimal data type for a pandas Series or DataFrame

    Parameters:
        data: pd.Series or pd.DataFrame
            Data to analyze
        force_int: bool, default False
            Forces the function to treat the series as integers

    Returns:
        If input is a Series, returns a numpy.dtype suggested based on the series values.
        If input is a DataFrame, returns a DataFrame with optimized data types.
    """

    def optimize_series(series):
        # Try converting to numeric if the series is of object type
        if not isinstance(series, pd.Series):
            return series
        
        if series.dtype == 'object':
            series = series.convert_dtypes().apply(pd.to_numeric, errors='ignore')
        
        # Drop NaN values for analysis
        series = series.dropna()

        if pd.api.types.is_integer_dtype(series) or force_int or (
            pd.api.types.is_float_dtype(series) and all(
            series.apply(lambda x: float(x).is_integer()))
        ):
            min_val, max_val = series.min(), series.max()
            if min_val >= 0:
                if max_val <= np.iinfo(np.uint8).max:
                    return series.astype(np.uint8)
                elif max_val <= np.iinfo(np.uint16).max:
                    return series.astype(np.uint16)
                elif max_val <= np.iinfo(np.uint32).max:
                    return series.astype(np.uint32)
                elif max_val <= np.iinfo(np.uint64).max:
                    return series.astype(np.uint64)
            else:
                if np.iinfo(np.int8).min <= min_val <= max_val <= np.iinfo(np.int8).max:
                    return series.astype(np.int8)
                elif np.iinfo(np.int16).min <= min_val <= max_val <= np.iinfo(np.int16).max:
                    return series.astype(np.int16)
                elif np.iinfo(np.int32).min <= min_val <= max_val <= np.iinfo(np.int32).max:
                    return series.astype(np.int32)
                elif np.iinfo(np.int64).min <= min_val <= max_val <= np.iinfo(np.int64).max:
                    return series.astype(np.int64)
        elif pd.api.types.is_float_dtype(series):
            min_val, max_val = series.min(), series.max()
            if np.finfo(np.float16).min <= min_val <= max_val <= np.finfo(np.float16).max:
                return series.astype(np.float16)
            elif np.finfo(np.float32).min <= min_val <= max_val <= np.finfo(np.float32).max:
                return series.astype(np.float32)
            elif np.finfo(np.float64).min <= min_val <= max_val <= np.finfo(np.float64).max:
                return series.astype(np.float64)
            elif np.finfo(np.longdouble).min <= min_val <= max_val <= np.finfo(np.longdouble).max:
                return series.astype(np.longdouble)
        else:
            return series

    if isinstance(data, pd.Series):
        return optimize_series(data)
    elif isinstance(data, pd.DataFrame):
        data.index = data.index.astype(int)
        for col in data.columns:
            data[col] = optimize_series(data[col])
        return data
    else:
        raise ValueError("Input must be a pandas Series or DataFrame")

def check_corrs(df, settings):
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    # Calculate the correlation matrix for numeric columns
    correlation_matrix = numeric_df.corr()
    # Focus on the correlation of all features with respect to target
    correlation_with_target = correlation_matrix[settings["target"]].sort_values(key=abs, ascending=False)
    # Take the absolute value of the correlation coefficients
    abs_correlation_with_target = correlation_with_target.abs()
    # Drop the target itself
    abs_correlation_with_target = abs_correlation_with_target.drop(settings["target"])
    # Sort by absolute value
    sorted_abs_correlation = abs_correlation_with_target.sort_values(ascending=False)
    # Select top 10 and bottom 10 features
    top_10_features = sorted_abs_correlation.head(10).index.tolist()
    bottom_10_features = sorted_abs_correlation.tail(10).index.tolist()
    # Combine them
    selected_features = top_10_features + bottom_10_features
    # Include target to the selected features for the heatmap
    selected_features.append(settings["target"])

    filtered_df = df
    if 0.0 < settings["corr_thresh"] < 1.0:
        # Drop columns based on the correlation threshold, keeping protected columns
        drop_columns = sorted_abs_correlation[sorted_abs_correlation < settings["corr_thresh"]].index.tolist()
        drop_columns = [col for col in drop_columns if col not in settings["protected_columns"]]
        filtered_df = df.drop(columns=drop_columns)
        print(f"Dropped {df.shape[1]-filtered_df.shape[1]} columns due to low feature correlation with target")
        print(f"New data shape: {filtered_df.shape}")
        
    if settings["plot_corrs"]:
        # Generate the heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(df[selected_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title(f'Correlation Heatmap of Top and Bottom Numeric Features with {settings["target"]}')
        plt.show()
        plt.figure(figsize=(10, 6))
        sns.histplot(abs_correlation_with_target, bins=20, kde=False, color='blue')
        plt.xlabel(f'Absolute Correlation with {settings["target"]}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Absolute Correlations with {settings["target"]}')
        plt.grid(True)
        plt.show()

    return filtered_df

def convert_to_percent_change(df, columns=None):
    """
    Convert specified DataFrame columns to percent change.

    Parameters:
    df (pandas.DataFrame): DataFrame to convert.
    columns (list): List of column names to convert.

    Returns:
    pandas.DataFrame: DataFrame with specified columns converted to percent change.
    """
    df = df.copy()
    if columns is None:
        columns = df.columns

    for column in columns:
        df[column + ' (%)'] = df[column].pct_change() * 100
    return df

def convert_back_to_original(percent_df, original_df, columns):
    """
    Convert specified DataFrame columns back to the original format for new dates.

    Parameters:
    percent_df (pandas.DataFrame): DataFrame in percent change form.
    original_df (pandas.DataFrame): Original DataFrame.
    columns (list): List of column names to convert.

    Returns:
    pandas.DataFrame: DataFrame with new values converted back and merged with the original.
    """
    # Copy the dataframes to avoid modifying the originals
    percent_df = percent_df.copy()
    original_df = original_df.copy()
    # Identify the latest date in the original dataframe
    latest_date = original_df['Datetime'].max()
    # Filter percent-change dataframe for rows with dates greater than the latest date in original
    new_values_df = percent_df[percent_df['Datetime'] > latest_date]
    # Convert percent-change values back to original form, starting from the latest known value
    for column in columns:
        # Calculate the last original value before the new dates
        last_value = original_df.loc[original_df['Datetime'] == latest_date, column].iloc[0]
        # Convert back using the last known value and percent change
        new_values_df[column] = last_value * (new_values_df[column + ' (%)'] / 100 + 1).cumprod()

    # Drop the percent change columns from the new values dataframe
    new_values_df = new_values_df.drop([col + ' (%)' for col in columns], axis=1)
    # Merge the newly converted values with the original dataframe
    merged_df = pd.concat([original_df, new_values_df])
    # Sort by date just in case
    merged_df.sort_values('Datetime', inplace=True)
    return merged_df

def best_model_path(settings: dict):
    """
    Find highest checkpoint to load.

    Returns:
    path (str): Path to the highest checkpoint.
    """
    # Define the path to the trial's checkpoint directory
    trials = pd.read_csv("model_performances.csv")
    sorted_trials = trials.sort_values(by=settings["loss_name"], inplace=True)
    return sorted_trials["ckpt_path"][0]

def save_model(model, trainer, settings, csv_path='model_performances.csv'):
    # Save model checkpoint
    torch.save(model.state_dict(), settings["model_path"])

    # Prepare model performance data
    model_performance = {metric_name: metric_value.item() if isinstance(
                        metric_value, torch.Tensor) else metric_value 
                         for metric_name, metric_value in trainer.callback_metrics.items()
    }
    model_performance.update({
        'lstm_layers': settings["lstm_layers"],
        'name': settings["name"],
        'corr_thresh': settings["corr_thresh"],
        'pct': settings["pct"],
        'float_precision': settings["float_precision"],
        'max_epochs': settings["max_train_epochs"],
        'limit_train_batches': settings["limit_train_batches"],
        'log_every_n_steps': settings["log_every_n_steps"],
        'reduce_on_plateau_patience': settings["reduce_on_plateau_patience"],
        'batch_size': settings["batch_size"],
        'target': settings["target"],
        'max_prediction_length': settings["max_prediction_length"],
        'train_test_pct': settings["train_test_pct"],
        'n_splits': settings["n_splits"],
        'scaling_method': settings["scaling_method"],
        'model_path': settings["model_path"],
        'ckpt_path': trainer.checkpoint_callback.best_model_path.replace("\\", "/"),
    })
    model_df = pd.DataFrame([model_performance])

    # Append to existing file or create new
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, model_df])
    else:
        combined_df = model_df

    # Remove duplicate rows based on all columns
    combined_df.drop_duplicates(inplace=True)

    # Reorder columns to have MAPE and RMSE first
    first_cols = [settings["loss_name"], settings["secondary_loss"]]
    other_cols = [col for col in combined_df.columns if col not in first_cols]
    combined_df = combined_df[first_cols + other_cols]

    # Sort by loss
    combined_df.sort_values(by=settings["loss_name"], inplace=True)

    # Save to CSV
    combined_df.to_csv(csv_path, index=False)

    mean_tft_mape = trainer.callback_metrics[settings["loss_name"]].item()
    
    if mean_tft_mape < settings["loss"]:
        settings["loss"] = mean_tft_mape
        # Update settings with model hyperparameters
        # Assuming hyperparameters are stored in model.hparams
        # Modify this part according to your model's structure
        for key, value in model.hparams.items():
            settings[key] = value

        with open("best_settings.json", "w") as f:
            json.dump(settings, f, indent=4, default=serialize)

def save_study(study, settings, csv_path='study_results.csv'):
    # Extract trial data
    trials_df = study.trials_dataframe(attrs=('number', 'user_attrs', 'duration', ))
    
    # Rename columns for clarity
    trials_df.columns = trials_df.columns.str.replace('user_attrs_', '', regex=True)

    # Reorder columns to have MAPE and RMSE first
    first_cols = [settings["loss_name"], settings["secondary_loss"]]
    other_cols = [col for col in trials_df.columns if col not in first_cols]
    trials_df = trials_df[first_cols + other_cols]

    # Append to existing file or create new
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, trials_df])
    else:
        combined_df = trials_df

    # Remove duplicate rows based on all columns
    combined_df.drop_duplicates(inplace=True)
    
    # Sort by MAPE
    combined_df.sort_values(by=settings["loss_name"], inplace=True)

    # Save to CSV
    combined_df.to_csv(csv_path, index=False)

    if not os.path.exists(settings["study_path"]):
        directory = os.path.dirname(settings["study_path"])
        os.makedirs(directory, exist_ok=True)

    with open(settings["study_path"], "wb") as fout:
        pickle.dump(study, fout)

def load_model_from_ckpt(settings: dict, file_path: str = None):
    # if no file_path passed, assume best
    if file_path is None:
        file_path = best_model_path(settings)

    if file_path.endswith(".ckpt"):
        return TemporalFusionTransformer.load_from_checkpoint(file_path)
    elif file_path.endswith(".pth"):
        settings = settings
        # Load and preprocess data
        original = pd.read_csv(settings["data"])
        # Preprocess data, settings and check correlations
        data, settings = preprocess(original, settings) 
        # Create dataset
        max_encoder_length = len(data) // settings['n_splits']
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
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=settings["learning_rate"],
            lstm_layers=settings["lstm_layers"],
            hidden_size=settings["hidden_size"],
            hidden_continuous_size=settings["hidden_continuous_size"],
            attention_head_size=settings["attention_head_size"],
            dropout=settings["dropout"],
            output_size=settings["output_size"],
            #loss=settings["loss_func"],
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="Ranger",
            reduce_on_plateau_patience=settings["reduce_on_plateau_patience"],
        )
        # Check if the file exists to continue training
        if os.path.exists(settings["model_path"]):
            checkpoint = torch.load(settings["model_path"])
            tft.load_state_dict(checkpoint["state_dict"])
            print("Study loaded successfully.")
        else:
            print(f"The file {settings['model_path']}.pth does not exist.")
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        return tft
    else:
        return TemporalFusionTransformer.load_from_checkpoint(best_model_path(settings))

def ipynb_to_py(path):
    with open(path+".ipynb", "r", encoding="utf-8") as f:
        notebook = read(f, as_version=4)
    
    # Initialize an empty string to store the Python code
    py_code = ""
    
    for cell in notebook.cells:
        if cell.cell_type == "code":
            py_code += ''.join(cell.source) + '\n\n'

        elif cell.cell_type == 'markdown':
            # Convert markdown to comments
            comments = '# ' + cell.source.replace('\n', '\n# ')
            py_code += comments + '\n\n'
            
    with open(path+".py", "w", encoding="utf-8") as f:
        f.write(py_code)

def handle_datetime_column(data, col):
    # Convert to datetime, handling errors
    dates = pd.to_datetime(data[col], errors='coerce')
    data = data.dropna(subset=[col])  # Drop rows with NaT in Datetime

    # Rank the dates to get a range of integers from 1 to n
    data[col] = dates.rank(method='dense').astype(int)

    # Extract datetime features based on inferred frequency
    freq_col = pd.DatetimeIndex(dates)
    inferred_freq = pd.infer_freq(freq_col)
    if isinstance(inferred_freq, str):
        data['Year'] = dates.dt.year
        data['Month'] = dates.dt.month
        if 'D' in inferred_freq:  # Daily data
            data['Day'] = dates.dt.day
            data['DayOfWeek'] = dates.dt.dayofweek
            data['DayOfYear'] = dates.dt.dayofyear
            data['WeekOfTheYear'] = dates.dt.isocalendar().week
        if 'H' in inferred_freq:  # Hourly data
            data['Hour'] = dates.dt.hour
    return data

def preprocess(df, settings, dt_to_int=True):
    """
    Process data to be trained.

    Parameters:
    df (pandas.DataFrame): DataFrame to process.
    settings (dict): Dictionary containing processing settings.

    Returns:
    pandas.DataFrame: Processed DataFrame.
    """
    data = df.copy()

    # Reset index to ensure it is an integer range
    data.reset_index(drop=True, inplace=True)

    # Convert to percent if specified
    if settings.get("pct"):
        data = convert_to_percent_change(data, [settings['target']])
        data.drop(settings['target'], axis=1, inplace=True)
        data.dropna(inplace=True)
        settings['target'] = settings['target'] + ' (%)'

    # Handle DateTime, String, and Numeric columns
    for col in data.columns:
        # Normalize numeric columns
        if pd.api.types.is_numeric_dtype(data[col]):
            # Apply log scaling or z-score normalization
            if settings.get("scaling_method") == 'log':
                data[col] = np.log1p(data[col])
            else:
                scaler = StandardScaler()
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        
        elif dt_to_int:
            # Note datetime columns to convert to ints
            if dt_to_int and pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = data[col].astype('int64').rank(method='dense').astype(int)
            # Attempt to check if string col has dates,
            # else encode by enumerating set of strings
            elif pd.api.types.is_string_dtype(data[col]) or data[col].dtype == "object":
                try:
                    temp = data[col].copy()

                    # Attempt to convert column to datetime
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    # Note down the column if conversion was successful
                    if not data[col].isnull().all():
                        data[col] = data[col].astype('int64').rank(method='dense').astype(int)
                    else:
                        # Enumerate unique strings
                        data[col], _ = pd.factorize(temp)
                        data[col] = optimize_data_types(data[col])

                except Exception as e:
                    print(f"Error converting column {col} to datetime: {e}")

    # Check correlations and drop columns if specified in settings
    if settings.get("corr_thresh", 0) > 0.0:
        data = check_corrs(data, settings)

    return data, settings

def serialize(obj):
    if isinstance(obj, torch.nn.Module):
        return repr(obj)
    else:
        return None
    
def load_settings(name: str = "settings.json", settings: dict = None):
    if settings is None:
        with open(name, 'r') as file:
            settings = json.load(file)

        if settings["best"]:
            with open("best_" + name, 'r') as file:
                settings = json.load(file)

        return settings

    loss_funcs = {
        "val_SMAPE": SMAPE(), 
        "val_MAE":   MAE(),
        "val_RMSE":  RMSE(),
        "val_MAPE":  MAPE(),
    }

    df = pd.read_csv(settings["data"], low_memory=True)
    settings["data_shape"] = df.shape
    settings["max_encoder_length"] = len(df)
    settings["max_encoder_length_range"] = (1, settings["max_encoder_length"])
    settings["loss_func"] = loss_funcs.get(settings["loss_name"], QuantileLoss())
    settings["base_path"] = f"config_history/lstm_layers_{settings['lstm_layers']}"
    settings["study_path"] = f"{settings['base_path']}/{settings['name']}/{settings['name']}.pkl" # .pkl
    settings["model_path"] = f"{settings['base_path']}/{settings['name']}/{settings['name']}.pth" # .pth
    settings["ckpt_path"] = f"{settings['base_path']}/{settings['name']}/model_ckpts" # .ckpt
    settings["protected_columns"] = ["Datetime", "Ticker", settings["target"]]

    return settings

def start_study(settings: dict = None):
    if settings is None:
        settings = load_settings()

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_CUDA_ALLOC_SYNC'] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision(settings["float_precision"]) # use 'high' for more accurate models
    torch.set_grad_enabled(True)

    # Find and Load Previous Study

    # Check if the file exists
    last_study = None
    if os.path.exists(settings['study_path']):
        with open(settings['study_path'], "rb") as f:
            last_study = pickle.load(f)
        print("Study loaded successfully.")

    # Load and preprocess data
    original = pd.read_csv(settings["data"], low_memory=False)

    # Preprocess data, settings and check correlations
    data, settings = preprocess(original, settings) 
    print(data)
    print(data.info())
    # Create dataset
    # Calculate the number of data points to be used for training based on train_test_pct
    num_train = int(len(data) * settings["train_test_pct"])

    # Use the calculated number to set the training cutoff
    training_cutoff = data.sort_values("Datetime").iloc[num_train]["Datetime"]
    unknown_reals = list(data.columns.drop(['Datetime', 'Ticker']))
    
    training = TimeSeriesDataSet(
        data[lambda x: x.Datetime <= training_cutoff],
        time_idx="Datetime",
        target=settings["target"], 
        group_ids=["Ticker"],
        min_encoder_length=settings["max_encoder_length"] // 2,
        max_encoder_length=settings["max_encoder_length"],
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
    
    # configure network with new trainer and callbacks
    early_stop_callback = EarlyStopping(
        monitor=settings["loss_name"],
        patience=10,
    )
    ckpt_callback = ModelCheckpoint(
        monitor=settings["loss_name"],
        dirpath=settings["ckpt_path"],
        filename='{val_loss:.3f}-{epoch:02d}',
        every_n_epochs=1,
    )
    lr_logger = LearningRateMonitor() 

    logger = TensorBoardLogger(
        save_dir=f"lightning_logs/lstm_layers_{settings['lstm_layers']}/", 
        name=settings["name"],
    )

    # Start Study
    study = optimize_hyperparameters(
        study=last_study,
        training=training,
        val_dataloader=val_dataloader,
        train_dataloader=train_dataloader,
        loss=settings["loss_func"],
        callbacks=[early_stop_callback, ckpt_callback, lr_logger],
        logger=logger,
        settings=settings,
    )

    # Save the trials
    save_study(study, settings)

    # show best hyperparameters
    best_trial = study.best_trial
    print(best_trial)

    previous_best_checkpoint = ckpt_callback.best_model_path

    print("Using model path ", previous_best_checkpoint)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("highest")
    
    # configure network with new trainer and callbacks
    early_stop_callback = EarlyStopping(
        monitor=settings["loss_name"],
        patience=10,
    )
    ckpt_callback = ModelCheckpoint(
        monitor=settings["loss_name"],
        dirpath=settings["ckpt_path"],
        filename='{val_loss:.3f}-{epoch:02d}',
        every_n_epochs=1,
    )
    lr_logger = LearningRateMonitor() 

    logger = TensorBoardLogger(
        save_dir=f"lightning_logs/lstm_layers_{settings['lstm_layers']}/", 
        name=settings["name"],
    )

    best_tft = TemporalFusionTransformer.load_from_checkpoint(previous_best_checkpoint)

    # Optionally, continue training for more epochs
    trainer = pl.Trainer(
        max_epochs=settings["max_train_epochs"],
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
    save_model(best_tft, trainer, settings)

    torch.cuda.empty_cache()
    return

def fetch_data(url, settings):
    response = requests.get(url, params=settings)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def save_to_parquet(data, symbol, dataset):
    df = pd.DataFrame(data)
    df.to_parquet(f"Data/{symbol}/{dataset}.parquet")

def create_combined_timeseries_dataset(base_path, common_freq='5min'):
    """
    Combines multiple Parquet files into a single time series DataFrame.

    Parameters:
        base_path: str
            Path to the base directory containing ticker folders.
        common_freq: str
            The common frequency to resample all datasets to.

    Returns:
        pd.DataFrame: A combined time series DataFrame.
    """
    all_data = []

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.parquet'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_parquet(file_path)
                elif file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path)
                else:
                    continue

                # Update min and max dates
                min_date = min(min_date, df.index.min())
                max_date = max(max_date, df.index.max())

                all_data.append(df)

            # Resample and align data to common date range
            aligned_data = []
            for df in all_data:
                df = df.reindex(pd.date_range(min_date, max_date, freq=common_freq), method='ffill')
                aligned_data.append(df)

            # Assuming the DataFrame has a DateTime index; if not, set it
            df.set_index('Datetime', inplace=True)

            # Resample or align the data
            df_resampled = df.resample(common_freq).ffill()  # Forward fill for coarser data

            all_data.append(df_resampled)

    # Combine all dataframes
    combined_df = pd.concat(all_data, axis=1)

    # Remove rows with any NaN values
    combined_df.dropna(inplace=True)

    return combined_df

def parse_response_data(data):
    """Parses the response data into a DataFrame and sets appropriate index."""
    
    if "Meta Data" in data:
        # Convert the data to a DataFrame and transpose it
        df = pd.DataFrame(data[list(data.keys())[1]]).T
        df.rename(columns=lambda col: re.sub(r'^\d+\.\s+', '', col).lower(), inplace=True)
        df["Datetime"] = pd.to_datetime(df.index)
    
    elif any(key.startswith('quarterly') for key in data) or 'annualEarnings' in data:
        key = next((k for k in data.keys() if k.startswith('quarterly')), 'annualEarnings')
        df = pd.DataFrame(data[key])
        df["Datetime"] = pd.to_datetime(df['fiscalDateEnding'])
        df.drop(columns=['fiscalDateEnding',], inplace=True)
    
    elif 'data' in data and all(k in ['date', 'value'] for k in data['data'][0].keys()):
        df = pd.DataFrame(data['data'])
        df["Datetime"] = pd.to_datetime(df['date'])
        df[data['name']] = pd.to_numeric(df["value"], errors='coerce')
        df.drop(columns=["value", "date"], inplace=True)
    
    else:
        raise ValueError("Unknown data structure in JSON response")

    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    return df.copy()