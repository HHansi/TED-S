# Created by Hansi at 12/28/2021

import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')

MODEL_TYPE = "bert"
MODEL_NAME = "bert-base-uncased"


config = {
    "best_model_dir": os.path.join(OUTPUT_DIRECTORY, "best_model"),
    'cache_dir': os.path.join(OUTPUT_DIRECTORY, "cache_dir"),
    'output_dir': OUTPUT_DIRECTORY,

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 84,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': True,

    'logging_steps': 200,  #40,
    'save_steps': 200,  #40,
    "no_cache": False,
    'save_model_every_epoch': True,
    "save_recent_only": True,
    # 'n_fold': 3,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 200,  #40,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,

    'regression': False,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,

    "manual_seed": SEED,

    "encoding": None,
    "sliding_window": False,

    "labels_list": [0, 1, 2],
}
