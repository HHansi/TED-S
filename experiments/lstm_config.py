# Created by Hansi at 12/22/2021
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')

SEED = 157

config = {
    'manual_seed': SEED,
    'model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),
    'best_model_path': os.path.join(OUTPUT_DIRECTORY, "model", "best_model"),

    # 'embedding_size': 300,
    'maxlen': 84,  # 72,  # max sequence length
    'max_features': None,  # how many unique words to use (i.e num rows in embedding vector)
    'batch_size': 64,  # 64, 8
    'n_epochs': 20,
    'earlystopping_patience': 5,  # 2
    'prediction_batch_size': 64,

    'train_file': 'train.tsv',
    'dev_file': 'dev.tsv',

    # 'embedding_file': 'F:/workspace-backup/Models/crawl-300d-2M-subword/crawl-300d-2M-subword.vec',
    # 'embedding_file': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec',
    # 'glove_embedding_file': 'F:/workspace-backup/Models/glove.840B.300d/glove.840B.300d.txt',

    'label_list': ['positive', 'negative', 'neutral'],

    # 'embedding_details': {'glove': 'F:/workspace-backup/Models/glove.840B.300d/glove.840B.300d.txt',
    #                       'fasttext': 'F:/workspace-backup/Models/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},
    'embedding_details': {'glove': '/content/glove.840B.300d/glove.840B.300d.txt',
                          'fasttext': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},

}