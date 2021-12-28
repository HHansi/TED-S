# Created by Hansi at 12/28/2021
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score

from algo.models.bert_classifier import ClassificationModel
from algo.models.common.evaluate import get_eval_results, macro_f1
from algo.models.common.label_encoder import encode, decode, reversed_label_mapping
from algo.util.data_preprocessor import preprocess_data
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments import transformer_config
from experiments.transformer_config import SEED, MODEL_NAME, MODEL_TYPE, PREDICTION_DIRECTORY, BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train(train_file_path, dev_split=0.1, test_file_path=None):
    new_data_dir = os.path.join(transformer_config.OUTPUT_DIRECTORY, f"data")
    delete_create_folder(transformer_config.OUTPUT_DIRECTORY)
    delete_create_folder(new_data_dir)

    # load, split and save data
    data = pd.read_csv(train_file_path, sep="\t", encoding="utf-8")
    y = data['label']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_split, random_state=SEED)
    train_index, test_index = next(sss.split(data, y))

    train = data.iloc[train_index]
    train = train[['tweet', 'label']]
    train = train.rename({'tweet': 'text'}, axis=1)
    train = train.rename({'label': 'labels'}, axis=1)

    dev = data.iloc[test_index]
    dev = dev[['tweet', 'label']]
    dev = dev.rename({'tweet': 'text'}, axis=1)
    dev = dev.rename({'label': 'labels'}, axis=1)

    # encode labels
    train = encode(train, label_column='labels')
    dev = encode(dev, label_column='labels')

    # preprocess data
    train['text'] = train['text'].apply(lambda x: preprocess_data(x))
    dev['text'] = dev['text'].apply(lambda x: preprocess_data(x))

    # train.to_csv(os.path.join(new_data_dir, transformer_config.config['train_file']), sep="\t", index=False)
    logger.info(f"Train instances: {train.shape[0]}")
    # dev.to_csv(os.path.join(new_data_dir, transformer_config.config['dev_file']), sep="\t", index=False)
    logger.info(f"Dev instances: {dev.shape[0]}")

    # train model
    logger.info(f"Training model...")
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_config.config,
                                use_cuda=torch.cuda.is_available(), num_labels=len(transformer_config.config['labels_list']))
    model.train_model(train, eval_df=dev, macro_f1=macro_f1, f1=f1_score, recall=recall_score,
                      precision=precision_score)

    model = ClassificationModel(MODEL_TYPE, transformer_config.config["best_model_dir"], args=transformer_config.config,
                                use_cuda=torch.cuda.is_available(), num_labels=len(transformer_config.config['labels_list']))

    # evaluate model
    if test_file_path is not None:
        logger.info(f"Evaluating model...")
        test_data = pd.read_csv(test_file_path, sep="\t", encoding="utf-8")
        logger.info(f"Test data: {test_data.shape}")
        test_data = test_data.rename({'tweet': 'text'}, axis=1)
        test_data['text'] = test_data['text'].apply(lambda x: preprocess_data(x))
        # get model predictions
        preds, raw_preds = model.predict(test_data['text'].tolist())
        # decode predicted labels
        preds = decode(preds)

        eval_results = get_eval_results(test_data['label'].tolist(), preds)
        logger.info(eval_results)

        print(f'preds: {preds}')
        print(f'raw preds: {raw_preds}')


def predict(data_file_path):
    create_folder_if_not_exist(PREDICTION_DIRECTORY, is_file_path=False)
    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    data = data.rename({'tweet': 'text'}, axis=1)
    data['text'] = data['text'].apply(lambda x: preprocess_data(x))

    model = ClassificationModel(MODEL_TYPE, transformer_config.config["best_model_dir"], args=transformer_config.config,
                                use_cuda=torch.cuda.is_available())
    preds, raw_preds = model.predict(data['text'].tolist())
    # decode predicted labels
    preds = decode(preds)

    eval_results = get_eval_results(data['label'].tolist(), preds)
    logger.info(eval_results)

    data['predictions'] = preds
    for i in reversed_label_mapping.keys():
        data[reversed_label_mapping[i]] = raw_preds[:, i]
    data['id'] = data['id'].apply(lambda x: str(x))  # save id as a str to avoid round off by excel
    data.to_excel(os.path.join(PREDICTION_DIRECTORY, f'{file_name}.xlsx'), sheet_name='Sheet1', index=False)


if __name__ == '__main__':
    train_file_path = os.path.join(BASE_PATH, 'data/fifa_2014/train.tsv')
    test_file_path = os.path.join(BASE_PATH, 'data/fifa_2014/test.tsv')
    train(train_file_path, test_file_path=test_file_path)