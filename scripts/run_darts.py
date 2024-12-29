import darts_tools as dt
from darts_models import ModelEvaluator
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

settings = {
    "name": "darts",
    "best": False,
    "pct": False,
    "optuna": True,
    "plot_corrs": False,
    "target": "Last Price",
    "loss_name": "val_MAPE",
    "secondary_loss": "val_RMSE",
    "base_path": "config_history",
    "data_path": "data.csv",
    "time_col_name": "Datetime",
    "group_col_name": "Ticker",
    "max_prediction_length": 90,
    "tuning_epochs": 100,
    "training_epochs": 10,
    "n_trials": 50,
    "n_splits": 5,
    "corr_thresh": 0.20,
    "train_test_pct": 0.75,
    "float_precision": ["medium", "high", "highest"][0],
    "validation": [None, "rolling"][0],
    "scaling_method": [MinMaxScaler, StandardScaler][-1],
    "models": {
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
                    'custom': {'future': [dt.encode_us_holidays],
                        'past': [dt.encode_us_holidays]},
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
    },
}

models = ModelEvaluator(settings)

# Simply training one model, for example
single_model = "TFTModel"
models.train(single_model)
models.plot_k_series(single_model, k=5, direction=0)

# Analyzing and comparing all models
models.run_all_models(tune=False)
models.compare_models()