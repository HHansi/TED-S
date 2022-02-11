# Created by Hansi at 12/22/2021
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(dict_embedding_details, word_index, max_features):
    """
    Load embeddings to a matrix
    If multiple embedding details are provided with dict_embedding_details, their concatenation will be used.
    For tokens which are unknown to some embedding models will use a vector of 0s as the embedding.

    :param dict_embedding_details: {name:file_path}
        dictionary of unique name to refer to the embedding and path to embedding model
    :param word_index: list
    :param max_features: int
    :return: matrix, int
        embedding matrix and final embedding size
    """

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    dict_embedding_indices = dict()
    dict_embedding_lengths = dict()
    for k, v in dict_embedding_details.items():
        logger.info(f'Loading embedding model - {k}')
        temp_index = dict(get_coefs(*o.split(" ")) for o in open(v, encoding='utf-8') if len(o)>100 and o.split(" ")[0] in word_index)
        temp_length = len(list(temp_index.values())[0])
        dict_embedding_lengths[k] = temp_length
        dict_embedding_indices[k] = temp_index

    total_embed_size = sum(list(dict_embedding_lengths.values()))
    embedding_matrix = np.zeros((max_features, total_embed_size))

    for word, i in word_index.items():
        dict_embedding_vectors = dict()
        if i >= max_features: continue

        not_found_count = 0
        for k, v in dict_embedding_indices.items():
            temp_vector = v.get(word)
            if temp_vector is None:
                not_found_count += 1
                temp_vector = np.zeros(dict_embedding_lengths[k])
            dict_embedding_vectors[k] = temp_vector

        if not_found_count < len(dict_embedding_indices.keys()):
            embedding_matrix[i] = np.concatenate(list(dict_embedding_vectors.values()))

    logger.info(f'Generated embedding matrix.')
    return embedding_matrix, total_embed_size

