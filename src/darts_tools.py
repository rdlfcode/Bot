import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from nbformat import read
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from darts.metrics.metrics import mae, mape, rmse, smape, ql
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts import concatenate
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    ARIMA, AutoARIMA, NaiveDrift, CatBoostModel, Croston, DLinearModel, FFT, FourTheta,
    ExponentialSmoothing, GlobalNaiveAggregate, KalmanForecaster, LightGBMModel, 
    LinearRegressionModel, NBEATSModel, NHiTSModel, NLinearModel, Prophet, 
    RandomForest, RegressionModel, RNNModel, StatsForecastAutoARIMA, 
    StatsForecastAutoCES, StatsForecastAutoETS, StatsForecastAutoTheta, BATS, TBATS, 
    TCNModel, TFTModel, TiDEModel, TransformerModel, TSMixerModel, VARIMA, XGBModel
)

def calculate_window_size(encoder_length, prediction_length, train_test_pct):
    min_val_size = encoder_length + 1  # Ensure validation series is longer than encoder length
    min_train_size = prediction_length + min_val_size

    # Calculate the minimum window size to satisfy both conditions
    window_size = min_train_size / train_test_pct + min_val_size / (1 - train_test_pct)

    return int(np.ceil(window_size))  # Round up to ensure minimum lengths are met

def calculate_step_size(data_length, window_size, num_sessions):
    usable_data_length = data_length - window_size  # Length available for sliding the window
    step_size = usable_data_length / (num_sessions - 1)  # Divide by (num_sessions - 1) to get 5 sessions
    return int(np.floor(step_size))  # Round down to ensure we don't exceed data length

def generate_covariates(stock, cov_type, settings, split, target=False):
    """
    Generate covariates for a stock time series.

    Args:
    stock (TimeSeries): The input stock time series.
    cov_type (str): 'past' or 'future' indicating the type of covariate.
    settings (dict): Dictionary containing settings for the generation.
    split (int, float): the split between training and test sets, either as 
                        a percentage of total data length to be trained on or
                        the number of data points to be trained on.

    Returns:
    TimeSeries: The concatenated covariates time series.
    """
    attributes = ['week', 'month', 'quarter']
    covariates = []

    if 0 < split < 1:
        train_split = int(len(stock) * split)
    elif split > 1:
        train_split = len(stock) - split

    train, val = stock[:train_split], stock[train_split:]
    
    if target:
        # Transforming both past covariates and target series
        scaler = Scaler()
        target_train_transformed = scaler.fit_transform(train[settings["target"]])
        target_val_transformed = scaler.transform(val[settings["target"]])

    if cov_type == 'future':
        # Generate future covariates  
        for attribute in attributes:
            cov_series = datetime_attribute_timeseries(
                stock.time_index,
                attribute=attribute,
                one_hot=False,
                add_length=settings["max_prediction_length"]
            )
            covariates.append(cov_series)

        combined_covariates = concatenate(covariates, axis=1)

    # Scaling covariates to normalize their scale.
    scaler = Scaler()
    combined_covariates = scaler.fit_transform(combined_covariates)

    return combined_covariates

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
        data.index = optimize_series(data.index.astype(int))
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
        print(f"New data shape: {filtered_df.shape}")
        print(f"Dropped {df.shape[1]-filtered_df.shape[1]} columns due to low feature correlation with target")
        
    if settings["plot"]:
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
    
def load_settings(name: str = "settings.json", settings: dict = None):
    if settings is None:
        with open(name, 'r') as file:
            settings = json.load(file)

        if settings["best"]:
            with open("best_" + name, 'r') as file:
                settings = json.load(file)

    loss_funcs = {
        "val_SMAPE": smape, 
        "val_MAE":   mae,
        "val_RMSE":  rmse,
        "val_MAPE":  mape,
    }

    df = pd.read_csv(settings["data"], low_memory=True)
    settings["data_shape"] = df.shape
    settings["max_encoder_length_range"] = (1, settings["max_encoder_length"])
    settings["loss_func"] = loss_funcs.get(settings["loss_name"], ql)
    settings["base_path"] = "config_history"
    settings["study_path"] = f"{settings['base_path']}/{settings['name']}/{settings['name']}.pkl" # .pkl
    settings["model_path"] = f"{settings['base_path']}/{settings['name']}/{settings['name']}.pth" # .pth
    settings["ckpt_path"] = f"{settings['base_path']}/{settings['name']}/model_ckpts" # .ckpt
    settings["protected_columns"] = ["Datetime", "Ticker"]
    settings["loss"] = 1000
    return settings

def generate_future_covariates(time_index, max_prediction_length, scaler, attributes=None):
    future_cov = [] 
    if attributes is None:
        attributes = ["week", "month", "quarter"]

    for attribute in attributes:
        temp_cov = datetime_attribute_timeseries(
            time_index, attribute=attribute, one_hot=False, add_length=max_prediction_length
        )
        future_cov.append(temp_cov)
    
    future_cov = concatenate(future_cov, axis=1)
    future_cov = future_cov.astype(np.float32)

    train_future_cov, val_future_cov = future_cov[:len(time_index)], future_cov[len(time_index):]
    tf_train_future_cov = scaler.fit_transform(train_future_cov)
    tf_val_future_cov = scaler.transform(val_future_cov)
    future_cov = concatenate([tf_train_future_cov, tf_val_future_cov])
    return future_cov

def encode_us_holidays(index, pred_len=0):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=index.min(), end=index.max()+pd.Timedelta(days=pred_len))
    return index.isin(holidays).astype(float)

def save(model, settings, csv_path='model_performances.csv'):
    # Save model checkpoint
    model.save(settings["model_path"])

    # Prepare model performance data
    model_df = pd.DataFrame({
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
        'ckpt_path': settings["ckpt_path"],
        'model': settings["model"],
    }.update(settings["evaluations"]))

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

    best = existing_df[settings["loss_name"]].min()

    if  settings["loss"] < best:
        with open("best_settings.json", "w") as f:
            json.dump(settings, f, indent=4, default=repr)

def plt_predictions(pred_series, val_series):
    # plot actual series
    plt.figure(figsize=(12, 9))

    split = min(len(val_series), -len(pred_series)*3)
    val_series[split:].plot(label='Actual Data')
    pred_series.plot(label='Predictions')

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()
    plt.show()

def serialize(obj):
    if isinstance(obj, torch.nn.Module):
        return repr(obj)
    else:
        return None

# def rolling_window():
#     def calculate_window_size(encoder_length, prediction_length, train_test_pct):
#         min_val_size = encoder_length + 1  # Ensure validation series is longer than encoder length
#         min_train_size = prediction_length + min_val_size

#         # Calculate the minimum window size to satisfy both conditions
#         window_size = min_train_size / train_test_pct + min_val_size / (1 - train_test_pct)

#         return int(np.ceil(window_size))  # Round up to ensure minimum lengths are met

#     def calculate_step_size(data_length, window_size, num_sessions):
#         usable_data_length = data_length - window_size  # Length available for sliding the window
#         step_size = usable_data_length / (num_sessions - 1)  # Divide by (num_sessions - 1) to get 5 sessions
#         return int(np.floor(step_size))  # Round down to ensure we don't exceed data length

#     window_size = calculate_window_size(self.models["max_encoder_length"], self.models["max_prediction_length"], self.models["train_test_pct"])
#     step_size = calculate_step_size(len(self.split_data[0]), window_size, self.models["n_splits"])

#     for window_start in range(0, len(self.split_data[0]) - window_size + 1, step_size):
#         window_end = window_start + window_size
#         stock_window = [stock[window_start:window_end] for stock in self.split_data]

#         fit_kwargs = self.get_fit_kwargs(model_name, self.prepare_fit_kwargs(stock_window))
#         fit_kwargs.update({'epochs': self.models["training_epochs"] // self.models["n_splits"]})

#         model.fit(**fit_kwargs)
    
model_info = {
    "ARIMA": {
        "model_class": ARIMA,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "p": 12, 
            "d": 1, 
            "q": 0, 
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                        'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                    'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                            'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                        'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
        },
        "hyper_params": {
            "p": [0, 1, 2, 4, 6, 8, 10],
            "d": [0, 1, 2],
            "q": [0, 1, 2],
            "seasonal_order": [(0, 0, 0, 0), (1, 1, 1, 12)],
            "trend": ["n", "c", "t", "ct"]
        },
    },
    "AutoARIMA": {
        "model_class": AutoARIMA,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "start_p": 0,
            "d": None,
            "start_q": 0,
            "max_p": 5,
            "max_d": 2,
            "max_q": 5,
            "start_P": 0,
            "D": None,
            "start_Q": 0,
            "max_P": 2,
            "max_D": 1,
            "max_Q": 2,
            "m": 12,
            "seasonal": True,
            "stationary": False,
            "information_criterion": 'aic',
            "alpha": 0.05,
            "test": 'kpss',
            "seasonal_test": 'ocsb',
            "stepwise": True,
            "n_jobs": 1,
            "start_params": None,
            "trend": None,
            "method": 'lbfgs',
            "maxiter": 50,
            "offset_test_args": None,
            "seasonal_test_args": None,
            "suppress_warnings": True,
            "error_action": 'trace',
            "trace": False,
            "random": False,
            "random_state": None,
            "n_fits": 10,
            "out_of_sample_size": 0,
            "scoring": 'mse',
            "scoring_args": None,
            "with_intercept": 'auto',
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'], 
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            }
        },
        "hyper_params": {
            "start_p": [0, 1, 2],
            "max_p": [3, 5, 7],
            "start_q": [0, 1, 2],
            "max_q": [3, 5, 7],
            "m": [12],  # Seasonality (e.g., monthly)
            "seasonal": [True, False],
            "trend": ["n", "c", "t", "ct"]
        },
    },
    "NaiveDrift": {
        "model_class": NaiveDrift,
        "supports_trainer_kwargs": False,
        "model_parameters": {},  # No model parameters
        "hyper_params": {}  # No grid search parameters
    },
    "CatBoostModel": {
        "model_class": CatBoostModel,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "lags": None,
            "lags_past_covariates": None,
            "lags_future_covariates": None,
            "output_chunk_length": 1,
            "output_chunk_shift": 0,
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'], 
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,
            "quantiles": None,
            "random_state": None,
            "multi_models": True,
            "use_static_covariates": True,
            # CatBoost-specific parameters (examples; add more as needed):
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "loss_function": 'RMSE'
        },
        "hyper_params": {
            "lags": [None, 3, 6, 12],
            "lags_past_covariates": [None, 3, 6, 12],
            "lags_future_covariates": [(0, 0), (1, 3), (1, 7)],
            "output_chunk_length": [1, 3, 6],
            "iterations": [500, 1000, 1500],
            "learning_rate": [0.01, 0.03, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5]
        }
    },
    "Croston": {
        "model_class": Croston,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "version": 'classic',  # Options: 'classic', 'optimized', 'sba', 'tsb'
            "alpha_d": None, 
            "alpha_p": None,
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'], 
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            }, 
        },
        "hyper_params": {
            "version": ['classic', 'optimized', 'sba'],
            "alpha_d": [0.1, 0.2, 0.3],  # Only for 'tsb' version
            "alpha_p": [0.1, 0.2, 0.3]   # Only for 'tsb' version 
        }
    },
    "DLinearModel": {
        "supports_trainer_kwargs": True,
        "model_class": DLinearModel,
        "model_parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "output_chunk_shift": 0,
            "shared_weights": False, 
            "kernel_size": 25,
            "const_init": True,
            "use_static_covariates": True,
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'], 
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            # Additional kwargs for PyTorch Lightning module, trainer, and TorchForecastingModel:
            "loss_fn": None,  # Default is MSELoss
            "likelihood": None,
            "torch_metrics": None,
            "optimizer_cls": None,  # Default is Adam
            "optimizer_kwargs": None,
            "lr_scheduler_cls": None,
            "lr_scheduler_kwargs": None,
            "batch_size": 32,
            "n_epochs": 100,
            "model_name": None,
            "work_dir": None, 
            "log_tensorboard": False,
            "nr_epochs_val_period": 1,
            "force_reset": False,
            "save_checkpoints": False, 
            "random_state": None,
            "pl_trainer_kwargs": None,
            "show_warnings": False 
        },
        "hyper_params": {
            "input_chunk_length": [12, 24, 36],
            "output_chunk_length": [6, 12, 18],
            "shared_weights": [True, False], 
            "kernel_size": [15, 25, 35],
            "batch_size": [16, 32, 64],
            "n_epochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.1]  # Assuming Adam optimizer
        }
    },
    "FFT": {
        "model_class": FFT,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "nr_freqs_to_keep": 10,
            "required_matches": None,
            "trend": None,  # Options: 'poly', 'exp', None
            "trend_poly_degree": 3
        },
        "hyper_params": {
            "nr_freqs_to_keep": [5, 10, 15, 20],
            "trend": ['poly', 'exp', None],
            "trend_poly_degree": [2, 3, 4]  # Only if trend is 'poly'
        }
    },
    "ExponentialSmoothing": {
        "model_class": ExponentialSmoothing,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "trend": None,  # Options: 'additive', 'multiplicative', None (ModelMode enum)
            "damped": False, 
            "seasonal": None,  # Options: 'additive', 'multiplicative', None (SeasonalityMode enum)
            "seasonal_periods": None, 
            "random_state": 0,
            "kwargs": None,  # Additional kwargs for statsmodels ExponentialSmoothing
            "fit_kwargs": None  # Additional kwargs for statsmodels ExponentialSmoothing.fit()
        },
        "hyper_params": {
            "trend": ['additive', 'multiplicative', None], 
            "damped": [True, False],
            "seasonal": ['additive', 'multiplicative', None],
            "seasonal_periods": [None, 4, 12]  # Adjust based on data frequency
        }
    },
    "GlobalNaiveAggregate": {
        "model_class": GlobalNaiveAggregate,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "output_chunk_shift": 0,
            "agg_fn": 'mean',  # Options: 'mean', 'sum', or custom aggregation function
            # Additional kwargs for PyTorch Lightning module and trainer:
            "batch_size": 32,
            "model_name": None,
            "work_dir": None,
            "log_tensorboard": False,
            "nr_epochs_val_period": 1,
            "force_reset": False,
            "random_state": None,
            "pl_trainer_kwargs": None,
            "show_warnings": False
        },
        "hyper_params": {
            "input_chunk_length": [12, 24, 36],
            "output_chunk_length": [6, 12, 18],
            "agg_fn": ['mean', 'sum']
        }
    },
    "KalmanForecaster": {
        "model_class": KalmanForecaster,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "dim_x": 1,  # Size of the Kalman filter state vector
            "kf": None,  # Optional nfoursid.kalman.Kalman instance 
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'], 
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            }, 
        },
        "hyper_params": {
            "dim_x": [1, 2, 3]  # Explore different state vector sizes
        }
    },
    "LightGBMModel": {
        "model_class": LightGBMModel,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "lags": None,
            "lags_past_covariates": None,
            "lags_future_covariates": None,
            "output_chunk_length": 1,
            "output_chunk_shift": 0,
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'],
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,  # Options: 'quantile', 'poisson'
            "quantiles": None, 
            "random_state": None,
            "multi_models": True,
            "use_static_covariates": True,
            "categorical_past_covariates": None,
            "categorical_future_covariates": None,
            "categorical_static_covariates": None, 
            # LightGBM-specific parameters (examples; add more as needed):
            "boosting_type": 'gbdt',  # Options: 'gbdt', 'dart', 'goss', 'rf'
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 100, 
            "subsample_for_bin": 200000,
            "objective": None,
            "class_weight": None,
            "min_split_gain": 0.0,
            "min_child_weight": 0.001,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": None,
            "n_jobs": -1,
            "importance_type": 'split',
            "**kwargs": None  # Additional kwargs for lightgbm.LGBMRegressor
        }, 
        "hyper_params": {
            "lags": [None, 3, 6, 12],
            "lags_past_covariates": [None, 3, 6, 12],
            "lags_future_covariates": [(0, 0), (1, 3), (1, 7)],
            "output_chunk_length": [1, 3, 6], 
            "boosting_type": ['gbdt', 'dart'],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200]
        }
    },
    "LinearRegressionModel": {
        "model_class": LinearRegressionModel,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "lags": None,
            "lags_past_covariates": None,
            "lags_future_covariates": None,
            "output_chunk_length": 1,
            "output_chunk_shift": 0, 
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                           'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                       'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute'],
                             'future': ['absolute']},
                'custom': {'future': [encode_us_holidays],
                           'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,  # Options: 'quantile', 'poisson'
            "quantiles": None,
            "random_state": None,
            "multi_models": True,
            "use_static_covariates": True, 
            "**kwargs": None  # Additional kwargs for sklearn.linear_model.LinearRegression (or Quantile/Poisson Regressor)
        },
        "hyper_params": {
            "lags": [None, 3, 6, 12],
            "lags_past_covariates": [None, 3, 6, 12],
            "lags_future_covariates": [(0, 0), (1, 3), (1, 7)],
            "output_chunk_length": [1, 3, 6]
        }
    },
    "NBEATSModel": {
        "model_class": NBEATSModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "output_chunk_shift": 0,
            "generic_architecture": True,
            "num_stacks": 30,
            "num_blocks": 1,
            "num_layers": 4,
            "layer_widths": 256,
            "expansion_coefficient_dim": 5,
            "trend_polynomial_degree": 2,
            "dropout": 0.0,
            "activation": 'ReLU',  # Options: 'ReLU', 'RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid'
            "add_encoders": {
                'cyclic': {'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute']},
                'custom': {'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            # Additional kwargs for PyTorch Lightning module, trainer, and TorchForecastingModel: 
            "loss_fn": None,  # Default is MSELoss
            "likelihood": None, 
            "torch_metrics": None,
            "optimizer_cls": None,  # Default is Adam
            "optimizer_kwargs": None,
            "lr_scheduler_cls": None,
            "lr_scheduler_kwargs": None,
            "use_reversible_instance_norm": False,
            "batch_size": 32,
            "n_epochs": 100,
            "model_name": None,
            "work_dir": None, 
            "log_tensorboard": False,
            "nr_epochs_val_period": 1,
            "force_reset": False,
            "save_checkpoints": False, 
            "random_state": None,
            "pl_trainer_kwargs": None,
            "show_warnings": False 
        },
        "hyper_params": {
            "input_chunk_length": [12, 24, 36],
            "output_chunk_length": [6, 12, 18],
            "generic_architecture": [True, False], 
            "num_stacks": [10, 20, 30],  # Only if generic_architecture is True
            "num_blocks": [1, 2],
            "num_layers": [2, 4, 6],
            "layer_widths": [128, 256, 512],
            "expansion_coefficient_dim": [3, 5, 7], # Only if generic_architecture is True
            "trend_polynomial_degree": [1, 2, 3],  # Only if generic_architecture is False
            "dropout": [0.0, 0.1, 0.2],
            "activation": ['ReLU', 'LeakyReLU'], 
            "batch_size": [16, 32, 64],
            "n_epochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.1]  # Assuming Adam optimizer
        }
    }, 
    "NHiTSModel": {
        "model_class": NHiTSModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "output_chunk_shift": 0, 
            "num_stacks": 3,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 512,
            "pooling_kernel_sizes": None,
            "n_freq_downsample": None,
            "dropout": 0.1, 
            "activation": 'ReLU',  # Options: 'ReLU', 'RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid'
            "MaxPool1d": True,
            "add_encoders": {
                'cyclic': {'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'past': ['week', 'month', 'quarter']},
                'position': {'past': ['absolute']},
                'custom': {'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            # Additional kwargs for PyTorch Lightning module, trainer, and TorchForecastingModel: 
            "loss_fn": None,  # Default is MSELoss
            "likelihood": None,
            "torch_metrics": None, 
            "optimizer_cls": None,  # Default is Adam
            "optimizer_kwargs": None, 
            "lr_scheduler_cls": None,
            "lr_scheduler_kwargs": None,
            "use_reversible_instance_norm": False,
            "batch_size": 32,
            "n_epochs": 100,
            "model_name": None,
            "work_dir": None,
            "log_tensorboard": False,
            "nr_epochs_val_period": 1,
            "force_reset": False,
            "save_checkpoints": False,
            "random_state": None,
            "pl_trainer_kwargs": None,
            "show_warnings": False
        },
        "hyper_params": {
            "input_chunk_length": [12, 24, 36],
            "output_chunk_length": [6, 12, 18],
            "num_stacks": [2, 3, 4], 
            "num_blocks": [1, 2],
            "num_layers": [2, 4], 
            "layer_widths": [256, 512],
            "dropout": [0.0, 0.1, 0.2],
            "activation": ['ReLU', 'LeakyReLU'],
            "batch_size": [16, 32, 64],
            "n_epochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.1]  # Assuming Adam optimizer 
        }
    },
    "NLinearModel": {
        "model_class": NLinearModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "output_chunk_shift": 0,
            "shared_weights": False,
            "const_init": True,
            "normalize": False,
            "use_static_covariates": True,
            "loss_fn": torch.nn.MSELoss(),
            "likelihood": None,
            "torch_metrics": None,
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": {'lr': 0.001},
            "lr_scheduler_cls": None,
            "lr_scheduler_kwargs": None,
            "use_reversible_instance_norm": False,
            "batch_size": 32,
            "n_epochs": 100,
            "model_name": None,
            "work_dir": None,
            "log_tensorboard": False,
            "nr_epochs_val_period": 1,
            "force_reset": False,
            "save_checkpoints": False,
            "random_state": None,
            "pl_trainer_kwargs": None,
            "show_warnings": False,
            "add_encoders": {
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [encode_us_holidays]},  # Assuming encode_us_holidays is defined elsewhere
                'transformer': Scaler(),  # Assuming Scaler is from darts
                'tz': 'CET'
            }
        },
        "hyper_params": {
            "input_chunk_length": [12, 24, 36],
            "output_chunk_length": [6, 12, 18],
            "batch_size": [16, 32, 64],
            "n_epochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.1]  # Assuming Adam optimizer
        }
    },
    "ProphetModel": {
        "model_class": Prophet,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "add_seasonalities": [
                {
                    "name": "daily",
                    "seasonal_periods": 24,
                    "fourier_order": 5,
                    "prior_scale": 10.0,
                    "mode": "additive"
                },
                {
                    "name": "weekly",
                    "seasonal_periods": 168,
                    "fourier_order": 10,
                    "prior_scale": 15.0,
                    "mode": "multiplicative"
                }
            ],
            "country_holidays": "US",
            "suppress_stdout_stderror": True,
            "add_encoders": {
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'future': ['relative']},
                'custom': {'future': [encode_us_holidays]},  # Assuming encode_us_holidays is defined elsewhere
                'transformer': Scaler(),  # Assuming Scaler is from darts
                'tz': 'CET'
            },
            "cap": 1000,
            "floor": 100,
            "prophet_kwargs": {
                "seasonality_mode": "additive",
                "daily_seasonality": False,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "changepoint_prior_scale": 0.05
            }
        },
        "hyper_params": {
            "fourier_order": [3, 5, 10],
            "prior_scale": [10.0, 20.0],
            "changepoint_prior_scale": [0.01, 0.1],
            "seasonality_mode": ["additive", "multiplicative"]
        }
    },
    "RandomForestModel": {
        "model_class": RandomForest,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "lags": None,  # This can be customized as needed
            "lags_past_covariates": None,  # This can be customized as needed
            "lags_future_covariates": None,  # This can be customized as needed
            "output_chunk_length": 1,
            "output_chunk_shift": 0,
            "add_encoders": {
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [encode_us_holidays]},  # Assuming encode_us_holidays is defined elsewhere
                'transformer': Scaler(),  # Assuming Scaler is from darts
                'tz': 'CET'
            },
            "n_estimators": 100,
            "max_depth": None,  # This can be customized as needed
            "multi_models": True,
            "use_static_covariates": True,
            "kwargs": {}  # Additional keyword arguments for RandomForestRegressor
        },
        "hyper_params": {
            "lags": [None, 3, 6, 12],
            "lags_past_covariates": [None, 3, 6, 12],
            "lags_future_covariates": [(0, 0), (1, 3), (1, 7)],
            "output_chunk_length": [1, 3, 6],
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20]
        }
    },
    "RegressionModel": {
        "model_class": RegressionModel,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "lags": None,  # This can be customized as needed
            "lags_past_covariates": None,  # This can be customized as needed
            "lags_future_covariates": None,  # This can be customized as needed
            "output_chunk_length": 1,
            "output_chunk_shift": 0,
            "add_encoders": {
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [encode_us_holidays]},  # Assuming encode_us_holidays is defined elsewhere
                'transformer': Scaler(),  # Assuming Scaler is from darts
                'tz': 'CET'
            },
            "model": None,  # Scikit-learn-like model; can be set to a specific regression model
            "multi_models": True,
            "use_static_covariates": True
        },
        "hyper_params": {
            "lags": [None, 3, 6, 12],
            "lags_past_covariates": [None, 3, 6, 12],
            "lags_future_covariates": [(0, 0), (1, 3), (1, 7)],
            "output_chunk_length": [1, 3, 6]
        }
    },
    "RNNModel": {
        "model_class": RNNModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 24,  # This needs to be set based on specific use case
            "model": 'RNN',  # Can be 'RNN', 'LSTM', or 'GRU'
            "hidden_dim": 25,
            "n_rnn_layers": 1,
            "dropout": 0.0,
            "training_length": 24,  # Must be larger than input_chunk_length
            "kwargs": {
                "loss_fn": torch.nn.MSELoss(),
                "optimizer_cls": torch.optim.Adam,
                "optimizer_kwargs": {'lr': 0.001},
                "lr_scheduler_cls": None,
                "lr_scheduler_kwargs": None,
                "batch_size": 32,
                "n_epochs": 100,
                "model_name": None,
                "work_dir": None,
                "log_tensorboard": False,
                "nr_epochs_val_period": 1,
                "force_reset": False,
                "save_checkpoints": False,
                "random_state": None,
                "pl_trainer_kwargs": None,
                "show_warnings": False,
                "add_encoders": {
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_us_holidays]},  # Assuming encode_us_holidays is defined elsewhere
                    'transformer': Scaler(),  # Assuming Scaler is from darts
                    'tz': 'CET'
                }
            }
        },
    "hyper_params": {
        "input_chunk_length": [12, 24, 36],
        "output_chunk_length": [6, 12, 18],
        "batch_size": [16, 32, 64],
        "n_epochs": [50, 100, 200],
        "learning_rate": [0.001, 0.01, 0.1]
        }
    },
    "StatsForecastAutoARIMA": {
        "model_class": StatsForecastAutoARIMA,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "autoarima_args": (1, 1, 1),  # Example: (p, d, q)
            "autoarima_kwargs": {
                "seasonal_order": (1, 1, 1, 12),  # Example: (P, D, Q, m)
            },
            "add_encoders": {
                'cyclic': {'future': ['week', 'month', 'quarter'],
                        'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                    'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                            'future': ['relative']},
                'custom': {'future': [encode_us_holidays],  # Assuming this is defined
                        'past': [encode_us_holidays]},
                'transformer': Scaler()  # Assuming Scaler is from darts
            },
        },
        "hyper_params": {
            "autoarima_args": [(0, 1, 1), (1, 1, 0), (2, 1, 0)],  # Varying (p, d, q)
            "autoarima_kwargs": {
                "seasonal_order": [(0, 1, 1, 12), (1, 1, 0, 12), (1, 1, 2, 12)],  # Varying (P, D, Q, m)
                "trend": ["n", "c", "t", "ct"]
            },
        },
    },
    "StatsForecastAutoCES": {
        "model_class": StatsForecastAutoCES,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "autoces_args": (),  # No positional arguments by default
            "autoces_kwargs": {
                "information_criterion": "aic",  # Example: choose information criterion
                "n_jobs": 1,  # Example: number of jobs to run in parallel
                "random_state": 0,  # Example: set random state for reproducibility 
            },
            "add_encoders": {  # (Optional) Add encoders as needed
                'cyclic': {'future': ['week', 'month', 'quarter'],
                        'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                    'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                            'future': ['relative']},
                'custom': {'future': [encode_us_holidays],  # Assuming this is defined
                        'past': [encode_us_holidays]},
                'transformer': Scaler()  # Assuming Scaler is from darts
            },
        },
        "hyper_params": {
            "autoces_kwargs": {
                "information_criterion": ["aic", "bic"],  # Explore different information criteria
                "n_jobs": [1, -1],  # Test different parallel job settings
                "random_state": [0, 42],  # Try different random states
            },
        },
    },
    "StatsForecastAutoETS": {
        "model_class": StatsForecastAutoETS,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "autoets_args": (),  # No positional arguments by default
            "autoets_kwargs": {
                "information_criterion": "aic",  # Example: choose information criterion
                "n_jobs": 1,  # Example: number of jobs to run in parallel
                "random_state": 0,  # Example: set random state for reproducibility
            },
            "add_encoders": {  # (Optional) Add encoders as needed
                'cyclic': {'future': ['week', 'month', 'quarter'],
                        'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                    'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                            'future': ['relative']},
                'custom': {'future': [encode_us_holidays],  # Assuming this is defined
                        'past': [encode_us_holidays]},
                'transformer': Scaler()  # Assuming Scaler is from darts
            },
        },
        "hyper_params": {
            "autoets_kwargs": {
                "information_criterion": ["aic", "bic"],  # Explore different information criteria
                "n_jobs": [1, -1],  # Test different parallel job settings
                "random_state": [0, 42],  # Try different random states
                "error": ["add", "mul"],  # Explore different error types
                "trend": ["add", "mul", "None"],  # Explore different trend types
                "damped": [True, False],  # Explore damped trend options
                "seasonal": ["add", "mul", "None"],  # Explore different seasonality types
                "seasonal_periods": [12],  # Adjust based on data frequency
            },
        },
    },
    "StatsForecastAutoTheta": {
        "model_class": StatsForecastAutoTheta,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "autotheta_args": (),  # No positional arguments by default
            "autotheta_kwargs": {
                "method": "auto",  # Example: set the method for selecting the best Theta model
                "use_mle": True,  # Example: whether to use maximum likelihood estimation
            },
            "add_encoders": {  # (Optional) Add encoders as needed
                'cyclic': {'future': ['week', 'month', 'quarter'],
                        'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                    'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                            'future': ['relative']},
                'custom': {'future': [encode_us_holidays],  # Assuming this is defined
                        'past': [encode_us_holidays]},
                'transformer': Scaler()  # Assuming Scaler is from darts
            },
        },
        "hyper_params": {
            "autotheta_kwargs": {
                "method": ["auto", "theta", "otm"],  # Explore different methods for model selection
                "use_mle": [True, False],  # Test whether to use MLE
            },
        },
    },
    "BATS": {
        "model_class": BATS,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "use_box_cox": None,  # Let the model decide whether to use Box-Cox transformation
            "box_cox_bounds": (0, 1),  # Set bounds for Box-Cox parameter
            "use_trend": None,  # Let the model decide whether to include a trend
            "use_damped_trend": None,  # Let the model decide whether to use damped trend
            "seasonal_periods": 'freq',  # Automatically infer seasonality from data frequency
            "use_arma_errors": True,  # Try to model residuals with ARMA for better fit
        },
        "hyper_params": {
            "use_box_cox": [True, False],  # Explore with and without Box-Cox transformation
            "use_trend": [True, False],  # Explore with and without trend
            "use_damped_trend": [True, False],  # Explore with and without damped trend
            "seasonal_periods": ['freq', 12],  # Try automatic and fixed seasonality (adjust 12 based on your data) 
            "use_arma_errors": [True, False],  # Explore with and without ARMA error modeling
        },
    },
    "TBATS": {
        "model_class": TBATS,
        "supports_trainer_kwargs": False,
        "model_parameters": {  # Same parameters as BATS, but allows for multiple seasonalities
            "use_box_cox": None,
            "box_cox_bounds": (0, 1),
            "use_trend": None,
            "use_damped_trend": None,
            "seasonal_periods": [12, 52],  # Example: monthly and yearly seasonality 
            "use_arma_errors": True,
        },
        "hyper_params": {  # Similar grid search as BATS, but explore multiple seasonalities
            "use_box_cox": [True, False],
            "use_trend": [True, False],
            "use_damped_trend": [True, False],
            "seasonal_periods": [[12], [52], [12, 52]],  # Try different combinations of seasonalities
            "use_arma_errors": [True, False],
        },
    },
    "TCNModel": {
        "model_class": TCNModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 24,  # Adjust based on your data and forecasting needs
            "output_chunk_length": 6,  # Adjust based on your desired prediction horizon
            "kernel_size": 3, 
            "num_filters": 3,
            "num_layers": 4,  # Example: number of convolutional layers
            "dilation_base": 2,
            "dropout": 0.2,
            "random_state": 0,  # For reproducibility
            "add_encoders": {  # Add encoders as needed
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [encode_us_holidays]},  # Assuming this is defined
                'transformer': Scaler(),  # Assuming Scaler is from darts
                'tz': 'CET'
            },
            "likelihood": None,  # Optionally specify a likelihood for probabilistic forecasts 
            "loss_fn": None,  # Optionally specify a loss function (defaults to MSELoss)
        },
        "hyper_params": {
            "input_chunk_length": [12, 24, 36],
            "output_chunk_length": [6, 12, 18],
            "kernel_size": [2, 3, 5],
            "num_filters": [2, 4, 8],
            "num_layers": [2, 3, 4],
            "dilation_base": [2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "batch_size": [16, 32, 64],
            "n_epochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.1],  # Assuming Adam optimizer
        },
    },
    "TFTModel": {
        "model_class": TFTModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 365, 
            "output_chunk_length": 90, 
            "hidden_size": 61, 
            "lstm_layers": 2, 
            "num_attention_heads": 11, 
            "dropout": 0.1257,
            "hidden_continuous_size": 23, 
            "add_relative_index": True, 
            "add_encoders": {  
                'cyclic': {'future': ['week', 'month', 'quarter'],
                    'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                        'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                    'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "feed_forward": "GatedResidualNetwork",
        },
        "hyper_params": {
            "hidden_size": [1, 100],
            "lstm_layers": [1, 4],
            "num_attention_heads": [1, 16], 
            "dropout": [0.1, 0.3], 
            "hidden_continuous_size": [1, 100],
            "batch_size": [24, 48, 96], 
            "n_epochs": [25, 50, 100], 
            "learning_rate": [0.0001, 1.0],
            "quantiles": [[0.1, 0.5, 0.9], [0.05, 0.5, 0.95]], 
            "feed_forward": ["GatedResidualNetwork", "GLU", "Bilinear"], 
        },
    },
    "FourTheta": {
        "model_class": FourTheta,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "theta": 2, 
            "seasonality_period": None,  # Let the model infer seasonality
            "season_mode": SeasonalityMode.MULTIPLICATIVE, 
            "model_mode": ModelMode.ADDITIVE,
            "trend_mode": TrendMode.LINEAR, 
            "normalization": True,
        },
        "hyper_params": {
            "theta": [1, 2, 3], 
            "season_mode": [SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.ADDITIVE, SeasonalityMode.NONE],
            "model_mode": [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE], 
            "trend_mode": [TrendMode.LINEAR, TrendMode.EXPONENTIAL],
            "normalization": [True, False], 
        },
    },
    "TiDEModel": {
        "model_class": TiDEModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 365, 
            "output_chunk_length": 90,  
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16, 
            "hidden_size": 128,
            "temporal_width_past": 4,
            "temporal_width_future": 4,
            "temporal_decoder_hidden": 32,
            "use_layer_norm": False,
            "dropout": 0.1,
            "use_static_covariates": False,  # Aligning with the settings
            "add_encoders": {  # Aligning with the existing encoder settings
                'cyclic': {'future': ['week', 'month', 'quarter'],
                    'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                        'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                    'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,  # Setting to None to make it deterministic 
            "loss_fn": None,  # Using the default loss function (MSELoss)
        },
        "hyper_params": {
            "input_chunk_length": [180, 365, 730],  # Exploring different input chunk lengths
            "output_chunk_length": [30, 90, 180],  # Exploring different output chunk lengths
            "num_encoder_layers": [1, 2, 3],
            "num_decoder_layers": [1, 2, 3],
            "decoder_output_dim": [8, 16, 32],
            "hidden_size": [64, 128, 256], 
            "temporal_width_past": [2, 4, 8],
            "temporal_width_future": [2, 4, 8],
            "temporal_decoder_hidden": [16, 32, 64],
            "use_layer_norm": [True, False],
            "dropout": [0.1, 0.2, 0.3],
            "batch_size": [24, 48, 96], 
            "n_epochs": [25, 50, 100],
            "learning_rate": [0.0001, 1.0],  # Assuming Adam optimizer
        },
    },
    "TransformerModel": {
        "model_class": TransformerModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 365, 
            "output_chunk_length": 90,  
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "activation": 'relu',
            "add_encoders": {  
                'cyclic': {'future': ['week', 'month', 'quarter'],
                    'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                        'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                    'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,  # Setting to None to make it deterministic 
            "loss_fn": None,  # Using the default loss function (MSELoss)
        },
        "hyper_params": {
            "input_chunk_length": [180, 365, 730],
            "output_chunk_length": [30, 90, 180], 
            "d_model": [32, 64, 128],
            "nhead": [2, 4, 8],
            "num_encoder_layers": [2, 3, 4],
            "num_decoder_layers": [2, 3, 4],
            "dim_feedforward": [256, 512, 1024],
            "dropout": [0.1, 0.2, 0.3],
            "activation": ['relu', 'gelu'], 
            "batch_size": [24, 48, 96], 
            "n_epochs": [25, 50, 100], 
            "learning_rate": [0.0001, 1.0],  
        },
    },
    "TSMixerModel": {
        "model_class": TSMixerModel,
        "supports_trainer_kwargs": True,
        "model_parameters": {
            "input_chunk_length": 365, 
            "output_chunk_length": 90,  
            "hidden_size": 64,
            "ff_size": 64, 
            "num_blocks": 2,
            "activation": 'ReLU',
            "dropout": 0.1,
            "use_static_covariates": False,  # Aligning with the settings
            "add_encoders": {  # Aligning with the existing encoder settings
                'cyclic': {'future': ['week', 'month', 'quarter'],
                    'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                        'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                    'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,  # Setting to None to make it deterministic 
            "loss_fn": None,  # Using the default loss function (MSELoss)
        },
        "hyper_params": {
            "input_chunk_length": [180, 365, 730],  
            "output_chunk_length": [30, 90, 180],  
            "hidden_size": [32, 64, 128],
            "ff_size": [32, 64, 128],
            "num_blocks": [1, 2, 3], 
            "activation": ['ReLU', 'LeakyReLU', 'GELU'],
            "dropout": [0.1, 0.2, 0.3],
            "batch_size": [24, 48, 96], 
            "n_epochs": [25, 50, 100],
            "learning_rate": [0.0001, 1.0],  
        },
    },
    "VARIMA": {
        "model_class": VARIMA,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "p": 12,  # Aligning with the existing p value for ARIMA
            "d": 1,  # Aligning with the existing d value for ARIMA
            "q": 0,  # Aligning with the existing q value for ARIMA
            "trend": None, 
            "add_encoders": {  # Aligning with the existing encoder settings
                'cyclic': {'future': ['week', 'month', 'quarter'],
                    'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                        'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                    'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
        },
        "hyper_params": {
            "p": [0, 1, 2, 4, 6, 8, 10],  # Aligning with the existing p range for ARIMA
            "d": [0, 1, 2],  # Aligning with the existing d range for ARIMA
            "q": [0, 1, 2],  # Aligning with the existing q range for ARIMA
            "trend": ["n", "c", "t", "ct"], 
        },
    },
    "XGBModel": {
        "model_class": XGBModel,
        "supports_trainer_kwargs": False,
        "supports_trainer_kwargs": False,
        "model_parameters": {
            "lags": None,
            "lags_past_covariates": None,
            "lags_future_covariates": None,
            "output_chunk_length": 90,  # Align with max_prediction_length
            "output_chunk_shift": 0,
            "add_encoders": { 
                'cyclic': {'future': ['week', 'month', 'quarter'],
                    'past': ['week', 'month', 'quarter']},
                'datetime_attribute': {'future': ['week', 'month', 'quarter'],
                                'past': ['week', 'month', 'quarter']},
                'position': {'past': ['relative'],
                        'future': ['relative']},
                'custom': {'future': [encode_us_holidays],
                    'past': [encode_us_holidays]},
                'transformer': Scaler()
            },
            "likelihood": None,  # Setting to None to make it deterministic 
            "random_state": 42,  # Aligning with the random_state used elsewhere 
            "n_estimators": 100,  # Example: number of boosting rounds
            "max_depth": 6,  # Example: maximum tree depth
            "learning_rate": 0.1,  # Example: learning rate
        },
        "hyper_params": {
            "lags": [None, 3, 6, 12], 
            "lags_past_covariates": [None, 3, 6, 12],
            "lags_future_covariates": [(0, 0), (1, 3), (1, 7)], 
            "output_chunk_length": [30, 90, 180], 
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
        },
    },
}