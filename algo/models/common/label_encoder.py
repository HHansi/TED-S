# Created by Hansi at 12/22/2021

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reversed_label_mapping = {value: key for (key, value) in label_mapping.items()}


def encode(df):
    df['label'] = df['label'].replace('negative', label_mapping['negative'])
    df['label'] = df['label'].replace('neutral', label_mapping['neutral'])
    df['label'] = df['label'].replace('positive', label_mapping['positive'])
    return df


def decode(labels):
    decoded_labels = [reversed_label_mapping[i] for i in labels]
    return decoded_labels
