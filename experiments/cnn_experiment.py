# Created by Hansi at 12/30/2021
import logging
import os

from sklearn.utils import shuffle

from algo.models.cnn_model import CNNModel
from algo.models.common.evaluate import get_eval_results
from algo.models.common.label_encoder import encode, decode, reversed_label_mapping
from algo.util.data_processor import split_data, preprocess_data
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments import cnn_config
import pandas as pd

from experiments.cnn_config import SEED, BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(train_file_paths, test_file_paths=None, predictions_folder=None):
    """

    :param train_file_paths:
        train file format - .tsv with columns [id, tweet, label]
    :param test_file_paths:
    :param predictions_folder:
    :return:
    """
    delete_create_folder(cnn_config.OUTPUT_DIRECTORY)
    new_data_dir = os.path.join(cnn_config.OUTPUT_DIRECTORY, f"data")
    create_folder_if_not_exist(new_data_dir)

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
    train, dev = split_data(data, SEED, label_column='labels', test_size=cnn_config.config['dev_size'])

    # encode labels
    train = encode(train, label_column='labels')
    dev = encode(dev, label_column='labels')

    # preprocess data
    train['text'] = train['text'].apply(lambda x: preprocess_data(x))
    dev['text'] = dev['text'].apply(lambda x: preprocess_data(x))

    train.to_csv(os.path.join(new_data_dir, cnn_config.config['train_file']), sep="\t", index=False)
    logger.info(f"Saved {train.shape[0]} train instances.")
    dev.to_csv(os.path.join(new_data_dir, cnn_config.config['dev_file']), sep="\t", index=False)
    logger.info(f"Saved {dev.shape[0]} dev instances.")

    # train model
    logger.info(f"Training model...")
    model = CNNModel('cnn', args=cnn_config.config, data_dir=new_data_dir)
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
            test_data['text'] = test_data['text'].apply(lambda x: preprocess_data(x))

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
    create_folder_if_not_exist(predictions_folder, is_file_path=False)
    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    data = data.rename({'tweet': 'text'}, axis=1)
    data['text'] = data['text'].apply(lambda x: preprocess_data(x))

    model = CNNModel(cnn_config.config['best_model_path'])
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
    # train_file_path = "F:/DataSets/Sentiment analysis/FIFA_2014_sentiment_dataset/data_100.tsv"

    fifa_train_file = os.path.join(BASE_PATH, 'data/fifa_2014/train.tsv')
    fifa_test_file = os.path.join(BASE_PATH, 'data/fifa_2014/test.tsv')
    semeval_train_file = os.path.join(BASE_PATH, 'data/semeval_data/train.tsv')
    semeval_test_file = os.path.join(BASE_PATH, 'data/semeval_data/test.tsv')
    munliv_train_file = os.path.join(BASE_PATH, 'data/munliv/munliv_train.tsv')
    munliv_test_file = os.path.join(BASE_PATH, 'data/munliv/munliv_test.tsv')
    predictions_folder = cnn_config.PREDICTION_DIRECTORY

    train_file_paths = [fifa_train_file]
    test_file_paths = [fifa_test_file, munliv_test_file]
    train(train_file_paths, test_file_paths=test_file_paths, predictions_folder=predictions_folder)

    munliv_annotation_file = os.path.join(BASE_PATH, 'data/munliv/munliv_annotations.tsv')
    predict(munliv_annotation_file, predictions_folder)
    brexitvote_annotation_file = os.path.join(BASE_PATH, 'data/brexitvote/brexitvote_annotations.tsv')
    predict(brexitvote_annotation_file, predictions_folder)
