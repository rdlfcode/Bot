import json
import torch
import darts_tools as dt

def main():
    settings = {
        "name": "testies",
        "best": False,
        "pct": False,
        "target": "Last Price",
        "loss_name": "val_MAPE",
        "secondary_loss": "val_RMSE",
        "n_splits": 5,
        "float_precision": ["medium", "high", "highest"][0],
        "plot_corrs": False,
        "validation": [None, "rolling"][-1],
        "lstm_layers": 2,
        "batch_size": 48,
        "max_study_epochs": 30,
        "max_train_epochs": 50,
        "n_trials": 1,
        "output_size": 90,
        "learning_rate": 0.001,
        "max_prediction_length": 90,
        "hidden_size": 61,
        "max_encoder_length": 365,
        "corr_thresh": 0.20,
        "train_test_pct": 0.75,
        "feed_forward": [
            "GatedResidualNetwork",
            "GLU",
            "Bilinear",
            "ReGLU",
            "GEGLU",
            "SwiGLU",
            "ReLU",
            "GELU"
        ][0],
        "gradient_clip_val_range": [
            0.01,
            1.0
        ],
        "lstm_layers_range": [
            1,
            4
        ],
        "hidden_size_range": [
            1,
            100
        ],
        "dropout_range": [
            0.1,
            0.3
        ],
        "hidden_continuous_size_range": [
            1,
            100
        ],
        "attention_head_size_range": [
            1,
            16
        ],
        "learning_rate_range": [
            0.0001,
            1.0
        ],
        "limit_train_batches": 30,
        "log_every_n_steps": 30,
        "reduce_on_plateau_patience": 10,
        "timeout": 604800,
        "loss": 0.53,
        "scaling_method": "z-score",
        "accelerator": 'cuda' if torch.cuda.is_available() else 'cpu',
        "data": "data.csv",
        "file": "model",
        "data_shape": [
            8259,
            84
        ],
        "max_encoder_length_range": [
            1,
            8259
        ],
        "loss_func": "MAPE()",
        "base_path": "config_history",
        "protected_columns": [
            "Datetime",
            "Ticker",
        ],
        "gradient_clip_val": 0.45,
        "dropout": 0.1257,
        "hidden_continuous_size": 23,
        "attention_head_size": 11,
        "model": "tft",
    }
    return settings

if __name__ == "__main__":
    settings = main()

    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4, default=dt.serialize)