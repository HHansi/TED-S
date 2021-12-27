# Created by Hansi at 12/22/2021
import logging
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# from algo.models.lstm_attention import LSTMModel
from algo.models.lstm import LSTMModel
from algo.util.data_preprocessor import preprocess_data
from algo.util.file_util import delete_create_folder
from experiments import lstm_config
from experiments.lstm_config import SEED, BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(train_file_path, dev_split=0.1, test_file_path=None):
    new_data_dir = os.path.join(lstm_config.OUTPUT_DIRECTORY, f"data")
    delete_create_folder(lstm_config.OUTPUT_DIRECTORY)
    delete_create_folder(new_data_dir)

    # load, split and save data
    data = pd.read_csv(train_file_path, sep="\t", encoding="utf-8")
    y = data['label']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_split, random_state=SEED)
    train_index, test_index = next(sss.split(data, y))

    train = data.iloc[train_index]
    dev = data.iloc[test_index]
    train = train.rename({'tweet': 'text'}, axis=1)
    dev = dev.rename({'tweet': 'text'}, axis=1)

    # preprocess data
    train['text'] = train['text'].apply(lambda x: preprocess_data(x))
    dev['text'] = dev['text'].apply(lambda x: preprocess_data(x))

    train.to_csv(os.path.join(new_data_dir, lstm_config.config['train_file']), sep="\t", index=False)
    logger.info(f"Saved {train.shape[0]} train instances.")
    dev.to_csv(os.path.join(new_data_dir, lstm_config.config['dev_file']), sep="\t", index=False)
    logger.info(f"Saved {dev.shape[0]} dev instances.")

    # train model
    logger.info(f"Training model...")
    model = LSTMModel(lstm_config.config)
    model.train(new_data_dir)

    # evaluate model
    if test_file_path is not None:
        logger.info(f"Evaluating model...")
        test_data = pd.read_csv(test_file_path, sep="\t", encoding="utf-8")
        logger.info(f"Test data: {test_data.shape}")
        test_data = test_data.rename({'tweet': 'text'}, axis=1)
        test_data['text'] = test_data['text'].apply(lambda x: preprocess_data(x))
        # get model predictions
        raw_preds, preds = model.predict(test_data['text'].tolist())

        eval_results = eval(test_data['label'].tolist(), preds)
        logger.info(eval_results)


def predict(data_file_path):
    # data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    # data = data.rename({'tweet': 'text'}, axis=1)
    # data['text'] = data['text'].apply(lambda x: preprocess_data(x))
    # texts = data['text'].tolist()

    # model = LSTMModel(lstm_config.config)
    # model.predict(texts)

    data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    data = data.rename({'tweet': 'text'}, axis=1)
    data['text'] = data['text'].apply(lambda x: preprocess_data(x))

    model = LSTMModel(lstm_config.config)
    raw_preds, preds = model.predict(data['text'].tolist())

    eval_results = eval(data['label'].tolist(), preds)
    logger.info(eval_results)


if __name__ == '__main__':
    # data_file_path = "F:/DataSets/Sentiment analysis/FIFA_2014_sentiment_dataset/data_100.tsv"
    train_file_path = os.path.join(BASE_PATH, 'data/fifa_2014/train.tsv')
    test_file_path = os.path.join(BASE_PATH, 'data/fifa_2014/test.tsv')
    train(train_file_path, test_file_path=test_file_path)

    # predict(test_file_path)


