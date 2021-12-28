# Created by Hansi at 12/22/2021

import re

from nltk import TweetTokenizer

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


def preprocess_data(text):
    text = text.replace("\n", " ")
    text = remove_links(text, substitute='')
    text = remove_retweet_notations(text)
    text = remove_repeating_characters(text)
    # tokenize and lower case
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    tokens = tknzr.tokenize(text)
    text = " ".join(tokens)
    # text.replace(symbol, "#")  # remove # in hash tags
    # remove white spaces at the beginning and end of the text
    text = text.strip()
    # remove extra whitespace, newline, tab
    text = ' '.join(text.split())
    return text
