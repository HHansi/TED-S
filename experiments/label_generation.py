# Created by Hansi at 1/14/2022
import os

import pandas as pd

from algo.util.label_encoder import decode


def calculate_final_class_probabilities(input_file_path, output_file_path):
    df = pd.read_excel(input_file_path, sheet_name="Sheet1")

    df_negative = df[["negative-lstm", "negative-cnn", "negative-bert"]]
    df['negative_mean'] = df_negative.mean(axis=1)
    df['negative_std'] = df_negative.std(axis=1)

    df_neutral = df[["neutral-lstm", "neutral-cnn", "neutral-bert"]]
    df['neutral_mean'] = df_neutral.mean(axis=1)
    df['neutral_std'] = df_neutral.std(axis=1)

    df_positive = df[["positive-lstm", "positive-cnn", "positive-bert"]]
    df['positive_mean'] = df_positive.mean(axis=1)
    df['positive_std'] = df_positive.std(axis=1)

    df_labels = df[['negative_mean', 'neutral_mean', 'positive_mean']]
    df['final_label'] = df_labels.idxmax(axis=1)
    df['final_label'] = df['final_label'].apply(lambda x: x.replace('_mean', ''))
    # df['final_label'] = decode(df_labels.idxmax(axis=1).tolist())

    df['id'] = df['id'].apply(lambda x: str(x))  # save id as a str to avoid round off by excel
    df.to_excel(output_file_path, sheet_name='Sheet1', index=False)


if __name__ == '__main__':
    input_file_path = 'data/brexitvote/predictions/brexitvote-08.00-13.59.xlsx'
    output_file_path = 'data/brexitvote/predictions/brexitvote-08.00-13.59-final.xlsx'
    calculate_final_class_probabilities(input_file_path, output_file_path)

