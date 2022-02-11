# Created by Hansi at 12/22/2021

import re

import demoji
from nltk import TweetTokenizer
from sklearn.model_selection import StratifiedShuffleSplit

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '..', '...', '…']


def remove_links(sentence, substitute=''):
    """
    Method to remove links in the given text
    parameters
    -----------
    :param sentence: str
    :param substitute: str
        which to replace link
    :return: str
        String without links
    """
    sentence = re.sub('https?:\/\/\S+', substitute, sentence, flags=re.MULTILINE)
    return sentence.strip()


def remove_repeating_characters(sentence):
    """
    remove non alphaneumeric characters which repeat more than 3 times by its 3 occurrence (e.g. ----- to ---)
    :param sentence:
    :return:
    """
    sentence = re.sub('(\W)\\1{3,}', '\\1', sentence)
    return sentence.strip()


def remove_retweet_notations(sentence):
    """
    Method to remove retweet notations in the given text
    parameters
    -----------
    :param sentence: str
    :return: str
        String without retweet notations
    """
    updated_sentence = re.sub(r'RT @[a-zA-Z0-9_/-]*:', '', sentence)
    return updated_sentence.strip()


def add_emoji_text(x):
    """
    Covert emoji to text
    :param x: str
    :return: str
        String where emojis are replaced by text
    """
    emoji_text = demoji.findall(x)
    for em in emoji_text.keys():
        x = x.replace(em, ' ' + emoji_text[em] + ' ')
    x = ' '.join(x.split())
    return x


def preprocess_data(text, preserve_case=False, emoji_to_text=False):
    """
    A Pipeline to preprocess data
    :param text: str
    :param preserve_case: boolean, optional
    :param emoji_to_text: boolean, optional
    :return: str
    """
    text = text.replace("\n", " ")
    text = remove_links(text, substitute='')
    text = remove_retweet_notations(text)
    text = remove_repeating_characters(text)
    if emoji_to_text:
       text = add_emoji_text(text)
    # tokenize and lower case
    tknzr = TweetTokenizer(preserve_case=preserve_case, reduce_len=True, strip_handles=False)
    tokens = tknzr.tokenize(text)
    text = " ".join(tokens)
    # text.replace(symbol, "#")  # remove # in hash tags
    # remove white spaces at the beginning and end of the text
    text = text.strip()
    # remove extra whitespace, newline, tab
    text = ' '.join(text.split())
    return text


def split_data(df, seed, label_column='label', test_size=0.1):
    """
    StratifiedShuffleSplit the given DataFrame
    :param df: DataFrame
    :param seed: int
    :param label_column: str
    :param test_size: float
    :return: DataFrame, DataFrame
        train and test
    """
    y = df[label_column]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(sss.split(df, y))

    train = df.iloc[train_index]
    test = df.iloc[test_index]
    return train, test

