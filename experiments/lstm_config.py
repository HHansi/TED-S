# Created by Hansi at 12/22/2021
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')

SEED = 157

config = {
    'manual_seed': SEED,
    'best_model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),

    'max_len': 96,  # max sequence length
    'max_features': None,  # how many unique words to use (i.e num rows in embedding vector)
    'num_train_epochs': 20,

    'train_batch_size': 64,
    'test_batch_size': 64,

    'early_stopping': True,
    'early_stopping_patience': 5,

    'learning_rate': 1e-3,

    'train_file': 'train.tsv',  # filename to save train data
    'dev_file': 'dev.tsv',  # filename to save dev data
    'dev_size': 0.1,

    'labels_list': ['positive', 'negative', 'neutral'],
    'emoji_to_text': False,

    # 'embedding_details': {'glove': 'F:/workspace-backup/Models/glove.840B.300d/glove.840B.300d.txt',
    #                       'fasttext': 'F:/workspace-backup/Models/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},
    'embedding_details': {'glove': '/content/glove.840B.300d/glove.840B.300d.txt',
                          'fasttext': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},
}
