# Created by Hansi at 12/22/2021

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reversed_label_mapping = {value: key for (key, value) in label_mapping.items()}


def encode(df, label_column='label'):
    # df[label_column] = df[label_column].replace('negative', label_mapping['negative'])
    # df[label_column] = df[label_column].replace('neutral', label_mapping['neutral'])
    # df[label_column] = df[label_column].replace('positive', label_mapping['positive'])

    df = df.copy()
    df.loc[df[label_column] == 'negative', label_column] = label_mapping['negative']
    df.loc[df[label_column] == 'neutral', label_column] = label_mapping['neutral']
    df.loc[df[label_column] == 'positive', label_column] = label_mapping['positive']
    return df


def decode(labels):
    decoded_labels = [reversed_label_mapping[i] for i in labels]
    return decoded_labels
