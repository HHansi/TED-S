# Created by Hansi at 12/30/2021

import logging
import os
import pickle

import pandas as pd
import tensorflow as tf
from keras.utils.tf_utils import set_random_seed
from numpy import argmax
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils

from algo.models.common.embedding_util import load_concatenated_embeddings
from algo.models.config.nn_model_args import ClassificationArgs
from algo.util.file_util import create_folder_if_not_exist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel:
    def __init__(self, model_type_or_path, data_dir=None, args=None):

        if os.path.isdir(model_type_or_path):
            self.args = self._load_model_args(model_type_or_path)
            set_random_seed(self.args.manual_seed)
            with open(os.path.join(self.args.model_dir, 'tokenizer.pickle'), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.model = load_model(model_type_or_path)

        elif args:
            self.args = ClassificationArgs()
            if isinstance(args, dict):
                self.args.update_from_dict(args)
            elif isinstance(args, ClassificationArgs):
                self.args = args
            set_random_seed(self.args.manual_seed)

            if data_dir is None:
                raise ValueError(f'data directory is not defined!')
            train_df = pd.read_csv(os.path.join(data_dir, self.args.train_file), sep="\t", encoding="utf-8")
            dev_df = pd.read_csv(os.path.join(data_dir, self.args.dev_file), sep="\t", encoding="utf-8")

            X_train = train_df['text'].tolist()
            y_train = train_df['labels'].tolist()
            X_dev = dev_df['text'].tolist()
            y_dev = dev_df['labels'].tolist()

            # create tokenizer
            self.tokenizer = Tokenizer(num_words=self.args.max_features, filters='')
            self.tokenizer.fit_on_texts(list(X_train))

            self.X_train = self._format_data(X_train)
            self.X_dev = self._format_data(X_dev)
            # convert integers to dummy variables (i.e. one hot encoded)
            self.y_train = np_utils.to_categorical(y_train)
            self.y_dev = np_utils.to_categorical(y_dev)

            word_index = self.tokenizer.word_index
            self.args.max_features = len(word_index) + 1

            embedding_matrix, embedding_size = load_concatenated_embeddings(self.args.embedding_details, word_index,
                                                                            self.args.max_features)

            # define model structure
            inp = tf.keras.Input(shape=(self.args.max_len,), dtype="int64", name="input")
            x = layers.Embedding(self.args.max_features, embedding_size,
                                 embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                                 name="embedding_layer")(inp)
            x = layers.Conv1D(128, 5, activation="relu")(x)
            x = layers.MaxPooling1D(5)(x)
            x = layers.Conv1D(128, 5, activation="relu")(x)
            x = layers.MaxPooling1D(5)(x)
            x = layers.Conv1D(128, 5, activation="relu")(x)
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(len(self.args.labels_list), activation="softmax", name="dense_predictions")(x)
            self.model = tf.keras.Model(inputs=inp, outputs=x, name="cnn1D_model")

            opt = keras.optimizers.Adam(learning_rate=self.args.learning_rate)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            logger.info(self.model.summary())

    def train(self):
        create_folder_if_not_exist(self.args.model_dir)

        # train_df = pd.read_csv(os.path.join(data_dir, self.args.train_file), sep="\t", encoding="utf-8")
        # dev_df = pd.read_csv(os.path.join(data_dir, self.args.dev_file), sep="\t", encoding="utf-8")
        #
        # X_train = train_df['text'].tolist()
        # y_train = train_df['labels'].tolist()
        # X_dev = dev_df['text'].tolist()
        # y_dev = dev_df['labels'].tolist()
        #
        # # convert integers to dummy variables (i.e. one hot encoded)
        # y_train = np_utils.to_categorical(y_train)
        # y_dev = np_utils.to_categorical(y_dev)
        #
        # # create and save tokenizer
        # self.tokenizer = Tokenizer(num_words=self.args.max_features, filters='')
        # self.tokenizer.fit_on_texts(list(X_train))
        # # self.tokenizer.fit_on_texts(X_train + X_dev)
        # with open(os.path.join(self.args.model_dir, 'tokenizer.pickle'), 'wb') as handle:
        #     pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # X_train = self.process_data(X_train)
        # X_dev = self.process_data(X_dev)
        #
        # word_index = self.tokenizer.word_index
        # self.args.max_features = len(word_index) + 1
        #
        # embedding_matrix, embedding_size = load_concatenated_embeddings(self.args.embedding_details, word_index,
        #                                                                 self.args.max_features)
        #
        # # define model structure
        # inp = tf.keras.Input(shape=(self.args.max_len,), dtype="int64", name="input")
        # x = layers.Embedding(self.args.max_features, embedding_size,
        #                      embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
        #                      name="embedding_layer")(inp)
        # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name="lstm_1"))(x)
        # x = layers.Bidirectional(layers.LSTM(64, name="lstm_2"))(x)
        # x = layers.Dense(256, activation="relu", name="dense_1")(x)
        # x = layers.Dense(len(self.args.labels_list), activation="softmax", name="dense_predictions")(x)
        # self.model = tf.keras.Model(inputs=inp, outputs=x, name="lstm_model")
        #
        # opt = keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        # self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # logger.info(self.model.summary())

        checkpoint = ModelCheckpoint(self.args.best_model_path, monitor='val_loss', verbose=2, save_best_only=True,
                                     mode='min')
        callbacks = [checkpoint]
        if self.args.reduce_lr_on_plateau:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.args.reduce_lr_on_plateau_factor,
                                          patience=self.args.reduce_lr_on_plateau_patience,
                                          min_lr=self.args.reduce_lr_on_plateau_min_lr, verbose=2)
            callbacks.append(reduce_lr)
        if self.args.early_stopping:
            earlystopping = EarlyStopping(monitor='val_loss', min_delta=self.args.early_stopping_min_delta,
                                          patience=self.args.early_stopping_patience, verbose=2, mode='auto')
            callbacks.append(earlystopping)

        self.model.fit(self.X_train, self.y_train, batch_size=self.args.train_batch_size,
                       epochs=self.args.num_train_epochs,
                       validation_data=(self.X_dev, self.y_dev), verbose=2,
                       callbacks=callbacks, )
        self.model.load_weights(self.args.best_model_path)

        # save model args and tokenizer
        self.args.save(self.args.model_dir)
        with open(os.path.join(self.args.model_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, texts):
        # if self.model is None:
        #     # load tokenizer
        #     with open(os.path.join(self.args.model_dir, 'tokenizer.pickle'), 'rb') as handle:
        #         self.tokenizer = pickle.load(handle)
        #     # load model
        #     self.model = load_model(self.args.best_model_path)

        X_test = self._format_data(texts)

        raw_y_pred = self.model.predict([X_test], batch_size=self.args.test_batch_size, verbose=2)
        y_pred = [argmax(y) for y in raw_y_pred]

        return y_pred, raw_y_pred

    def _format_data(self, X):
        if self.tokenizer is None:
            raise ValueError('No Tokenizer found!')
        # tokenize the sequences
        X = self.tokenizer.texts_to_sequences(X)
        # pad the sentences
        X = pad_sequences(X, maxlen=self.args.max_len, padding='post', truncating='post')
        return X

    @staticmethod
    def _load_model_args(input_dir):
        args = ClassificationArgs()
        args.load(input_dir)
        return args
