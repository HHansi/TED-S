# Created by Hansi at 5/25/2022
import logging
import os

import pandas as pd
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

from algo.util.data_processor import preprocess_data
from algo.util.evaluate import get_eval_results
from algo.util.file_util import create_folder_if_not_exist

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')
PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sid = SentimentIntensityAnalyzer()


def get_sentiment(text):
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment


def predict(data_file_path, predictions_folder, evaluate=True):
    create_folder_if_not_exist(PREDICTION_DIRECTORY, is_file_path=False)
    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    data = pd.read_csv(data_file_path, sep="\t", encoding="utf-8")
    data = data.rename({'tweet': 'text'}, axis=1)
    data['text'] = data['text'].apply(
        lambda x: preprocess_data(x, preserve_case=True, emoji_to_text=False, remove_stopwords=True))

    preds = []
    for text in tqdm(data['text'].tolist()):
        preds.append(get_sentiment(text))

    if evaluate:
        eval_results = get_eval_results(data['label'].tolist(), preds)
        logger.info(eval_results)

    data['predictions'] = preds
    data['id'] = data['id'].apply(lambda x: str(x))  # save id as a str to avoid round off by excel
    data.to_excel(os.path.join(predictions_folder, f'{file_name}.xlsx'), sheet_name='Sheet1', index=False)


if __name__ == '__main__':
    fifa_train_file = os.path.join(BASE_PATH, 'data_private/fifa_2014/train.tsv')
    fifa_test_file = os.path.join(BASE_PATH, 'data_private/fifa_2014/test.tsv')
    semeval_train_file = os.path.join(BASE_PATH, 'data_private/semeval_data/train.tsv')
    semeval_test_file = os.path.join(BASE_PATH, 'data_private/semeval_data/test.tsv')
    munliv_train_file = os.path.join(BASE_PATH, 'data_private/munliv/munliv_train.tsv')
    munliv_test_file = os.path.join(BASE_PATH, 'data_private/munliv/munliv_test.tsv')
    brexitvote_train_file = os.path.join(BASE_PATH, 'data_private/brexitvote/brexitvote_train.tsv')
    brexitvote_test_file = os.path.join(BASE_PATH, 'data_private/brexitvote/brexitvote_test.tsv')
    predictions_folder = PREDICTION_DIRECTORY

    predict(brexitvote_test_file, predictions_folder, evaluate=True)
