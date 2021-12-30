# Created by Hansi at 12/22/2021
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')

SEED = 157

config = {
    'manual_seed': SEED,
    'best_model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),
    # 'best_model_path': os.path.join(OUTPUT_DIRECTORY, "model", "lstm_weights_best.h5"),  # lstm_attention_weights_best.h5

    'max_len': 84,  # 72, 96  # max sequence length
    'max_features': None,  # how many unique words to use (i.e num rows in embedding vector)
    'num_train_epochs': 20,

    'train_batch_size': 64,  # 64, 8
    'test_batch_size': 64,

    'early_stopping': True,
    'early_stopping_patience': 5,  # 2

    'learning_rate': 1e-3,

    'train_file': 'train.tsv',  # filename to save train data
    'dev_file': 'dev.tsv',  # filename to save dev data
    'dev_size': 0.1,

    # 'embedding_file': 'F:/workspace-backup/Models/crawl-300d-2M-subword/crawl-300d-2M-subword.vec',
    # 'embedding_file': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec',
    # 'glove_embedding_file': 'F:/workspace-backup/Models/glove.840B.300d/glove.840B.300d.txt',

    'labels_list': ['positive', 'negative', 'neutral'],

    # 'embedding_details': {'glove': 'F:/workspace-backup/Models/glove.840B.300d/glove.840B.300d.txt',
    #                       'fasttext': 'F:/workspace-backup/Models/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},
    # 'embedding_details': {'glove': '/content/glove.840B.300d/glove.840B.300d.txt',
    #                       'fasttext': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},
    # 'embedding_details': {'glove': '/home/rgcl-dl/Projects/TED-S/embedding_models/glove.840B.300d/glove.840B.300d.txt',
    #                       'fasttext': '/home/rgcl-dl/Projects/TED-S/embedding_models/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},

    'embedding_details': {'fasttext': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},

}