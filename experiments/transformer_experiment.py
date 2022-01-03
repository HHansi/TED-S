# Created by Hansi at 12/28/2021
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.utils import shuffle

from algo.util.evaluate import get_eval_results, macro_f1, macro_recall, macro_precision
from algo.util.label_encoder import encode, decode, reversed_label_mapping
from algo.models.transformer.transformer_model import ClassificationModel
from algo.util.data_processor import preprocess_data, split_data
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments import transformer_config
from experiments.transformer_config import SEED, MODEL_NAME, MODEL_TYPE, BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train(train_file_paths, test_file_paths=None, predictions_folder=None):
    """

    :param train_file_paths:
        train file format - .tsv with columns [id, tweet, label]
    :param test_file_paths:
    :param predictions_folder:
    :return:
    """
    delete_create_folder(transformer_config.OUTPUT_DIRECTORY)
    new_data_dir = os.path.join(transformer_config.OUTPUT_DIRECTORY, f"data")
    delete_create_folder(new_data_dir)

    # merge training data
    data = pd.DataFrame(columns=['id', 'tweet', 'label'])
    for path in train_file_paths:
        temp_data = pd.read_csv(path, sep="\t", encoding="utf-8")
        data = data.append(temp_data, ignore_index=True)
    if len(train_file_paths) > 1:
        data = shuffle(data, random_state=SEED)

    # format data
    data = data[['tweet', 'label']]
    data = data.rename({'tweet': 'text'}, axis=1)
    data = data.rename({'label': 'labels'}, axis=1)

    # split training data
    train, dev = split_data(data, SEED, label_column='labels', test_size=transformer_config.config['dev_size'])

    # encode labels
    train = encode(train, label_column='labels')
    dev = encode(dev, label_column='labels')

    # preprocess data
    preserve_case = not transformer_config.config['do_lower_case']
    train['text'] = train['text'].apply(lambda x: preprocess_data(x, preserve_case=preserve_case))
    dev['text'] = dev['text'].apply(lambda x: preprocess_data(x, preserve_case=preserve_case))

    logger.info(f"Train instances: {train.shape[0]}")
    logger.info(f"Dev instances: {dev.shape[0]}")

    # train model
    logger.info(f"Training model...")
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_config.config,
                                use_cuda=torch.cuda.is_available(),
                                num_labels=len(transformer_config.config['labels_list']))
    model.train_model(train, eval_df=dev, macro_f1=macro_f1, macro_recall=macro_recall, macro_precision=macro_precision)

    model = ClassificationModel(MODEL_TYPE, transformer_config.config["best_model_dir"], args=transformer_config.config,
                                use_cuda=torch.cuda.is_available(),
                                num_labels=len(transformer_config.config['labels_list']))

    # evaluate model
    if test_file_paths is not None:
        logger.info(f"Evaluating model...")
        for test_path in test_file_paths:
            logger.info(f'Predicting {test_path}...')
            test_data = pd.read_csv(test_path, sep="\t", encoding="utf-8")
            logger.info(f"Test data: {test_data.shape}")

            # format and preprocess data
            test_data = test_data.rename({'tweet': 'text'}, axis=1)
            test_data['text'] = test_data['text'].apply(lambda x: preprocess_data(x, preserve_case=preserve_case))

            # get model predictions
            preds, raw_preds = model.predict(test_data['text'].tolist())
            # decode predicted labels
            preds = decode(preds)
            # convert raw predictions to probabilities
            raw_preds_probabilities = softmax(raw_preds, axis=1)

            # evaluate results
            eval_results = get_eval_results(test_data['label'].tolist(), preds)
            logger.info(f'{test_path} results: {eval_results}')

            if predictions_folder is not None:
                create_folder_if_not_exist(predictions_folder, is_file_path=False)
                # save predictions
                file_name = os.path.splitext(os.path.basename(test_path))[0]
                test_data['predictions'] = preds
                for i in reversed_label_mapping.keys():
                    test_data[reversed_label_mapping[i]] = raw_preds_probabilities[:, i]
                test_data['id'] = test_data['id'].apply(
                    lambda x: str(x))  # save id as a str to avoid round off by excel
                test_data.to_excel(os.path.join(predictions_folder, f'{file_name}.xlsx'), sheet_name='Sheet1',
                                   index=False)


def predict(data_file_path, predictions_folder, evaluate=True):
    create_folder_if_not_exist(predictions_folder, is_file_path=False)
    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    data = data.rename({'tweet': 'text'}, axis=1)
    preserve_case = not transformer_config.config['do_lower_case']
    data['text'] = data['text'].apply(lambda x: preprocess_data(x, preserve_case=preserve_case))

    model = ClassificationModel(MODEL_TYPE, transformer_config.config["best_model_dir"], args=transformer_config.config,
                                use_cuda=torch.cuda.is_available(),
                                num_labels=len(transformer_config.config['labels_list']))
    preds, raw_preds = model.predict(data['text'].tolist())
    # decode predicted labels
    preds = decode(preds)
    # convert raw predictions to probabilities
    raw_preds_probabilities = softmax(raw_preds, axis=1)

    if evaluate:
        eval_results = get_eval_results(data['label'].tolist(), preds)
        logger.info(eval_results)

    data['predictions'] = preds
    for i in reversed_label_mapping.keys():
        data[reversed_label_mapping[i]] = raw_preds_probabilities[:, i]
    data['id'] = data['id'].apply(lambda x: str(x))  # save id as a str to avoid round off by excel
    data.to_excel(os.path.join(predictions_folder, f'{file_name}.xlsx'), sheet_name='Sheet1', index=False)


if __name__ == '__main__':
    fifa_train_file = os.path.join(BASE_PATH, 'data/fifa_2014/train.tsv')
    fifa_test_file = os.path.join(BASE_PATH, 'data/fifa_2014/test.tsv')
    semeval_train_file = os.path.join(BASE_PATH, 'data/semeval_data/train.tsv')
    semeval_test_file = os.path.join(BASE_PATH, 'data/semeval_data/test.tsv')
    munliv_train_file = os.path.join(BASE_PATH, 'data/munliv/munliv_train.tsv')
    munliv_test_file = os.path.join(BASE_PATH, 'data/munliv/munliv_test.tsv')
    predictions_folder = transformer_config.PREDICTION_DIRECTORY

    train_file_paths = [fifa_train_file]
    test_file_paths = [fifa_test_file, munliv_test_file, semeval_test_file]
    train(train_file_paths, test_file_paths=test_file_paths, predictions_folder=predictions_folder)

    munliv_annotation_file = os.path.join(BASE_PATH, 'data/munliv/munliv_annotations.tsv')
    predict(munliv_annotation_file, predictions_folder)
    brexitvote_annotation_file = os.path.join(BASE_PATH, 'data/brexitvote/brexitvote_annotations.tsv')
    predict(brexitvote_annotation_file, predictions_folder)
