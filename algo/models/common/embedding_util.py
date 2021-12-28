# Created by Hansi at 12/22/2021
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fasttext(embedding_file, word_index, max_features):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding='utf-8') if len(o)>100 and o.split(" ")[0] in word_index )

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    total_embed_size = len(list(embeddings_index.values())[0])

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, total_embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix, total_embed_size


# def load_glove_fasttext(glove_embedding_file, fasttext_embedding_file, word_index, max_features, embed_size=600):
#     def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
#     glove_embedding_index = dict(get_coefs(*o.split(" ")) for o in open(glove_embedding_file, encoding='utf-8') if o.split(" ")[0] in word_index)
#     fasttext_embedding_index = dict(get_coefs(*o.split(" ")) for o in open(fasttext_embedding_file, encoding='utf-8') if len(o)>100 and o.split(" ")[0] in word_index )
#
#     embedding_matrix = np.zeros((max_features, embed_size))
#
#     for word, i in word_index.items():
#         if i >= max_features: continue
#         glove_embedding_vector = glove_embedding_index.get(word)
#         fasttext_embedding_vector = fasttext_embedding_index.get(word)
#
#         if glove_embedding_vector is None and fasttext_embedding_vector is None:
#             continue
#         else:
#             if glove_embedding_vector is None:
#                 glove_embedding_vector = np.zeros(300)
#             if fasttext_embedding_vector is None:
#                 fasttext_embedding_vector = np.zeros(300)
#             embedding_matrix[i] = np.concatenate([glove_embedding_vector, fasttext_embedding_vector])
#
#     return embedding_matrix


def load_concatenated_embeddings(dict_embedding_details, word_index, max_features):
    """

    :param dict_embedding_details: {name:file_path}
    :param word_index:
    :param max_features:
    :return:
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



if __name__ == '__main__':
    # a = np.zeros(300)
    # b = np.zeros(300)
    # c = np.concatenate([a,b])
    # print(np.zeros(300))

    test_dict = {'a':300, 'b':300}
    total_embed_size = sum(list(test_dict.values()))
    print(total_embed_size)