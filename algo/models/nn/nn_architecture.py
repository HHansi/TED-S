# Created by Hansi at 12/30/2021

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LSTM:
    def __init__(self, args, embedding_matrix):
        inp = tf.keras.Input(shape=(args.max_len,), dtype="int64", name="input")
        x = layers.Embedding(args.max_features, args.embedding_size,
                             embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                             name="embedding_layer")(inp)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name="lstm_1"))(x)
        x = layers.Bidirectional(layers.LSTM(64, name="lstm_2"))(x)
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dense(len(args.labels_list), activation="softmax", name="dense_predictions")(x)
        self.model = tf.keras.Model(inputs=inp, outputs=x, name="lstm_model")


class CNN1D:
    def __init__(self, args, embedding_matrix):
        inp = tf.keras.Input(shape=(args.max_len,), dtype="int64", name="input")
        x = layers.Embedding(args.max_features, args.embedding_size,
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
        x = layers.Dense(len(args.labels_list), activation="softmax", name="dense_predictions")(x)
        self.model = tf.keras.Model(inputs=inp, outputs=x, name="cnn1D_model")


class CNN2D:
    def __init__(self, args, embedding_matrix=None):
        filter_sizes = [1, 2, 3, 5]
        num_filters = 32

        inp = tf.keras.Input(shape=(args.max_len,), dtype="int64", name="input")
        x = layers.Embedding(args.max_features, args.embedding_size,
                             embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                             name="embedding_layer")(inp)

        x = layers.SpatialDropout1D(0.4, name="spatial_dropout_layer")(x)
        x = layers.Reshape((args.max_len, args.embed_size, 1), name="reshape_layer")(x)

        conv_0 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv0_layer")(x)
        conv_1 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv1_layer")(x)
        conv_2 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv2_layer")(x)
        conv_3 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[3], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv3_layer")(x)

        maxpool_0 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[0] + 1, 1), name="pool0_layer")(conv_0)
        maxpool_1 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[1] + 1, 1), name="pool1_layer")(conv_1)
        maxpool_2 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[2] + 1, 1), name="pool2_layer")(conv_2)
        maxpool_3 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[3] + 1, 1), name="pool3_layer")(conv_3)

        z = layers.Concatenate(axis=1, name="conc_layer")([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        z = layers.Flatten(name="flatten_layer")(z)
        z = layers.Dropout(0.1, name="dropout_layer")(z)
        outp = layers.Dense(args.num_classes, activation="softmax", name="dense_predictions")(z)
        self.model = tf.keras.Model(inputs=inp, outputs=outp, name="cnn_model")
