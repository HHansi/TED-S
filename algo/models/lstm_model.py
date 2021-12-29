# Created by Hansi at 12/22/2021
import logging
import os
import pickle

import pandas as pd
import tensorflow as tf
from keras.utils.tf_utils import set_random_seed
from numpy import argmax
from tensorflow import keras
from tensorflow.keras import layers
# import keras.backend as K
# from keras.layers import Dense
# from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils

from algo.models.common.embedding_util import load_concatenated_embeddings
from algo.models.common.nn_utils import Attention
from algo.models.config.lstm_model_args import ClassificationArgs
from algo.util.file_util import delete_create_folder

# import keras.backend as K
# from keras import regularizers, constraints
# from tensorflow import initializers
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from tensorflow.python.keras.layers import Embedding, LSTM, Bidirectional, Dense
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.saving.save import load_model
# from tensorflow.python.keras.utils import np_utils
# from tensorflow.python.layers.base import Layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, args):
        self.tokenizer = None
        self.model = None

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        set_random_seed(self.args.manual_seed)

    def process_data(self, X):
        if self.tokenizer is None:
            raise ValueError('No Tokenizer found!')
        # tokenize the sequences
        X = self.tokenizer.texts_to_sequences(X)
        # pad the sentences
        X = pad_sequences(X, maxlen=self.args.max_len, padding='post', truncating='post')
        return X

    def train(self, data_dir):
        delete_create_folder(self.args.model_dir)

        train_df = pd.read_csv(os.path.join(data_dir, self.args.train_file), sep="\t", encoding="utf-8")
        dev_df = pd.read_csv(os.path.join(data_dir, self.args.dev_file), sep="\t", encoding="utf-8")

        # # encode labels
        # train_df = encode(train_df)
        # dev_df = encode(dev_df)

        X_train = train_df['text'].tolist()
        y_train = train_df['labels'].tolist()
        X_dev = dev_df['text'].tolist()
        y_dev = dev_df['labels'].tolist()

        # convert integers to dummy variables (i.e. one hot encoded)
        y_train = np_utils.to_categorical(y_train)
        y_dev = np_utils.to_categorical(y_dev)

        # create and save tokenizer
        self.tokenizer = Tokenizer(num_words=self.args.max_features, filters='')
        self.tokenizer.fit_on_texts(list(X_train))
        # self.tokenizer.fit_on_texts(X_train + X_dev)
        with open(os.path.join(self.args.model_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        X_train = self.process_data(X_train)
        X_dev = self.process_data(X_dev)

        word_index = self.tokenizer.word_index
        self.args.max_features = len(word_index) + 1

        # embedding_matrix, embedding_size = load_fasttext(self.args['embedding_file'], word_index, self.args['max_features'])
        embedding_matrix, embedding_size = load_concatenated_embeddings(self.args.embedding_details, word_index,
                                                                        self.args.max_features)

        # define model structure
        # K.clear_session()
        # inp = Input(shape=(self.args['maxlen'],))
        # x = Embedding(self.args['max_features'], embedding_size, weights=[embedding_matrix], trainable=False)(inp)
        # x = Bidirectional(LSTM(64, return_sequences=True))(x)
        # x = Bidirectional(LSTM(64))(x)
        # # x = Bidirectional(LSTM(64, return_sequences=True))(x)
        # # x = Attention(self.args['maxlen'])(x)
        # x = Dense(256, activation="relu")(x)
        # # x = Dropout(0.25)(x)
        # x = Dense(len(self.args['label_list']), activation="softmax")(x)
        # self.model = Model(inputs=inp, outputs=x)

        inp = tf.keras.Input(shape=(None,), dtype="int64", name="input")
        x = layers.Embedding(self.args.max_features, embedding_size,
                             embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                             name="embedding_layer")(inp)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name="lstm_1"))(x)
        x = layers.Bidirectional(layers.LSTM(64, name="lstm_2"))(x)
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dense(len(self.args.labels_list), activation="softmax", name="dense_predictions")(x)
        self.model = tf.keras.Model(inputs=inp, outputs=x, name="lstm_model")

        opt = keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        logger.info(self.model.summary())

        # best_weights_path = os.path.join(self.args['model_dir'], 'lstm_attention_weights_best.h5')
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

        # callbacks = [checkpoint, reduce_lr, earlystopping]

        self.model.fit(X_train, y_train, batch_size=self.args.train_batch_size, epochs=self.args.num_train_epochs,
                       validation_data=(X_dev, y_dev), verbose=2,
                       callbacks=callbacks, )
        self.model.load_weights(self.args.best_model_path)

    def predict(self, texts):
        if self.model is None:
            # load tokenizer
            with open(os.path.join(self.args.model_dir, 'tokenizer.pickle'), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            # load model
            # best_weights_path = os.path.join(self.args['model_dir'], 'lstm_attention_weights_best.h5')
            self.model = load_model(self.args.best_model_path, custom_objects={'Attention': Attention})

        X_test = self.process_data(texts)

        raw_y_pred = self.model.predict([X_test], batch_size=self.args.test_batch_size, verbose=2)
        y_pred = [argmax(y) for y in raw_y_pred]

        # preds = decode(y_pred)

        return y_pred, raw_y_pred
