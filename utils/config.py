from time import strftime


configure = {
    "lr": {
        "type": float,
        "default": 1e-4
    },
    "ext_gamma": {
        "type": float,
        "default": 0.99
    },
    "int_gamma": {
        "type": float,
        "default": 0.9
    },
    "tau": {
        "type": float,
        "default": 1.0
    },
    "beta": {
        "type": float,
        "default": 0.01
    },
    "epsilon": {
        "type": float,
        "default": 0.2
    },
    "ext_coef": {
        "type": float,
        "default": 2.0
    },
    "int_coef": {
        "type": float,
        "default": 1.0
    },
    "update_proportion": {
        "type": float,
        "default": 0.25
    },
    "batch_size": {
        "type": int,
        "default": 128
    },
    "num_epochs": {
        "type": int,
        "default": 10
    },
    "local_steps": {
        "type": int,
        "default": 512
    },
    "global_steps": {
        "type": int,
        "default": 5e6
    },
    "save_interval": {
        "type": int,
        "default": 25
    },
    "model_path": {
        "type": str,
        "default": "trained_models"
    },
    "saved_path": {
        "type": str,
        "default": "trained_models/%s"%strftime("%Y-%m-%d-%H-%M-%S")
    },
    "tensorboard_path": {
        "type": str,
        "default": "tensorboard"
    },
    "record_path": {
        "type": str,
        "default": "records"
    },
    "log_path": {
        "type": str,
        "default": "log"
    }
}
