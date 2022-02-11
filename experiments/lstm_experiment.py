# Created by Hansi at 12/22/2021
import logging
import os

import pandas as pd
from sklearn.utils import shuffle

from algo.models.nn.nn_model import NNModel
from algo.util.evaluate import get_eval_results
from algo.util.label_encoder import reversed_label_mapping, encode, decode
from algo.util.data_processor import preprocess_data, split_data
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments import lstm_config
from experiments.lstm_config import SEED, BASE_PATH, PREDICTION_DIRECTORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(train_file_paths, test_file_paths=None, predictions_folder=None):
    """
    Train LSTM model
    :param train_file_paths: list
        train file format - .tsv with columns [id, tweet, label]
    :param test_file_paths: list, optional
        Given test file paths, the trained model will be evaluated on them and the results will be logged.
    :param predictions_folder: str, optional
        Given a predictions folder, test predictions will be saved
    :return:
    """
    delete_create_folder(lstm_config.OUTPUT_DIRECTORY)
    new_data_dir = os.path.join(lstm_config.OUTPUT_DIRECTORY, f"data")
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
    train, dev = split_data(data, SEED, label_column='labels', test_size=lstm_config.config['dev_size'])

    # encode labels
    train = encode(train, label_column='labels')
    dev = encode(dev, label_column='labels')

    # preprocess data
    train['text'] = train['text'].apply(lambda x: preprocess_data(x, preserve_case=False, emoji_to_text=lstm_config.config['emoji_to_text']))
    dev['text'] = dev['text'].apply(lambda x: preprocess_data(x, preserve_case=False, emoji_to_text=lstm_config.config['emoji_to_text']))

    train.to_csv(os.path.join(new_data_dir, lstm_config.config['train_file']), sep="\t", index=False)
    logger.info(f"Saved {train.shape[0]} train instances.")
    dev.to_csv(os.path.join(new_data_dir, lstm_config.config['dev_file']), sep="\t", index=False)
    logger.info(f"Saved {dev.shape[0]} dev instances.")

    # train model
    logger.info(f"Training model...")
    model = NNModel('lstm', data_dir=new_data_dir, args=lstm_config.config)
    model.train()

    # evaluate model
    if test_file_paths is not None:
        logger.info(f"Evaluating model...")
        for test_path in test_file_paths:
            logger.info(f'Predicting {test_path}...')
            test_data = pd.read_csv(test_path, sep="\t", encoding="utf-8")
            logger.info(f"Test data: {test_data.shape}")

            # format and preprocess data
            test_data = test_data.rename({'tweet': 'text'}, axis=1)
            test_data['text'] = test_data['text'].apply(lambda x: preprocess_data(x, preserve_case=False, emoji_to_text=lstm_config.config['emoji_to_text']))

            # get model predictions
            preds, raw_preds = model.predict(test_data['text'].tolist())
            # decode predicted labels
            preds = decode(preds)

            # evaluate results
            eval_results = get_eval_results(test_data['label'].tolist(), preds)
            logger.info(f'{test_path} results: {eval_results}')

            if predictions_folder is not None:
                create_folder_if_not_exist(predictions_folder, is_file_path=False)
                # save predictions
                file_name = os.path.splitext(os.path.basename(test_path))[0]
                test_data['predictions'] = preds
                for i in reversed_label_mapping.keys():
                    test_data[reversed_label_mapping[i]] = raw_preds[:, i]
                test_data['id'] = test_data['id'].apply(
                    lambda x: str(x))  # save id as a str to avoid round off by excel
                test_data.to_excel(os.path.join(predictions_folder, f'{file_name}.xlsx'), sheet_name='Sheet1',
                                   index=False)


def predict(data_file_path, predictions_folder, evaluate=True):
    """
    Predict using a model, and save final sentiment and confidence values to .xlsx file
    :param data_file_path: str
        format - .tsv file with column 'tweet'
    :param predictions_folder: str
    :param evaluate: boolean, optional
        If true the predictions will be evaluated and there should be a 'label' column in input data to use with
        evaluation.
    :return:
    """
    create_folder_if_not_exist(PREDICTION_DIRECTORY, is_file_path=False)
    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    data = data.rename({'tweet': 'text'}, axis=1)
    data['text'] = data['text'].apply(lambda x: preprocess_data(x, preserve_case=False, emoji_to_text=lstm_config.config['emoji_to_text']))

    model = NNModel(lstm_config.config['best_model_dir'])
    preds, raw_preds = model.predict(data['text'].tolist())
    # decode predicted labels
    preds = decode(preds)

    if evaluate:
        eval_results = get_eval_results(data['label'].tolist(), preds)
        logger.info(eval_results)

    data['predictions'] = preds
    for i in reversed_label_mapping.keys():
        data[reversed_label_mapping[i]] = raw_preds[:, i]
    data['id'] = data['id'].apply(lambda x: str(x))  # save id as a str to avoid round off by excel
    data.to_excel(os.path.join(predictions_folder, f'{file_name}.xlsx'), sheet_name='Sheet1', index=False)


if __name__ == '__main__':
    fifa_train_file = os.path.join(BASE_PATH, 'data/fifa_2014/train.tsv')
    fifa_test_file = os.path.join(BASE_PATH, 'data/fifa_2014/test.tsv')
    semeval_train_file = os.path.join(BASE_PATH, 'data/semeval_data/train.tsv')
    semeval_test_file = os.path.join(BASE_PATH, 'data/semeval_data/test.tsv')
    munliv_train_file = os.path.join(BASE_PATH, 'data/munliv/munliv_train.tsv')
    munliv_test_file = os.path.join(BASE_PATH, 'data/munliv/munliv_test.tsv')
    brexitvote_train_file = os.path.join(BASE_PATH, 'data/brexitvote/brexitvote_train.tsv')
    brexitvote_test_file = os.path.join(BASE_PATH, 'data/brexitvote/brexitvote_test.tsv')
    predictions_folder = lstm_config.PREDICTION_DIRECTORY

    train_file_paths = [fifa_train_file]
    test_file_paths = [fifa_test_file, munliv_test_file, semeval_test_file, brexitvote_test_file]
    # train(train_file_paths, test_file_paths=test_file_paths, predictions_folder=predictions_folder)

    munlive_file = os.path.join(BASE_PATH, 'data/munliv/munliv-15.28-17.23.tsv')
    predict(munlive_file, predictions_folder, evaluate=False)
    munlive_file_no_dups = os.path.join(BASE_PATH, 'data/munliv/munliv-15.28-17.23-no_duplicates.tsv')
    predict(munlive_file_no_dups, predictions_folder, evaluate=False)

    brexitvote_file = os.path.join(BASE_PATH, 'data/brexitvote/brexitvote-08.00-13.59.tsv')
    predict(brexitvote_file, predictions_folder, evaluate=False)
    brexitvote_file_no_dups = os.path.join(BASE_PATH, 'data/brexitvote/brexitvote-08.00-13.59-no_duplicates.tsv')
    predict(brexitvote_file_no_dups, predictions_folder, evaluate=False)