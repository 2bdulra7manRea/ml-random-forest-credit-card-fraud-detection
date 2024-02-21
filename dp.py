from datasets import load_dataset
import numpy as np


# {'id': [], 'index': [], 'time_elapsed': [], 'cc_num': [], 'amt': [], 'lat': [], 'long': []}

#  features: ['id', 'index', 'time_elapsed', 'cc_num', 'amt', 'lat', 'long', 'is_fraud']
#   num_rows: 2085138



SIMPLE_COUNT = 2085137


def get_credit_card_data():
    credit_card_data = load_dataset(
        "tanzuhuggingface/creditcardfraudtraining", split="train")
    return credit_card_data


def get_and_reshape_column_items(feature_name, data_set, simple_count):
    feature_column = data_set.select_columns(
        feature_name)[:simple_count][feature_name]
    return feature_column


def get_features_matrix(features_name, data_set, simple_count):
    table = []
    for column_name in features_name:
        items = get_and_reshape_column_items(
            column_name, data_set, simple_count)
        table.append(items)
    return np.array(table).reshape(simple_count, len(features_name))


def get_label_data(label_name, data_set, simple_count):
    return get_and_reshape_column_items(label_name, data_set, simple_count)


def x_y_dataset():
    credit_card_data = get_credit_card_data()
    features_name = ['long', 'lat', 'amt', 'time_elapsed']
    label_name = 'is_fraud'
    x = get_features_matrix(features_name, credit_card_data, SIMPLE_COUNT)
    y = get_label_data(label_name, credit_card_data, SIMPLE_COUNT)
    return x, y
