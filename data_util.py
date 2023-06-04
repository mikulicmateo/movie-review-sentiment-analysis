from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

import sys
np.set_printoptions(threshold=sys.maxsize)
def load_data_into_CountVector():
    with open("positive.txt", "r") as file:
        positive_lines = file.readlines()
    with open("negative.txt", "r") as file:
        negative_lines = file.readlines()    

    lines = np.append(positive_lines, negative_lines)
    count_vec = CountVectorizer(binary=True)
    count_matrix = count_vec.fit_transform(lines)
    tokens = count_matrix.toarray()

    # dictionary = []
    # with open("vocab.txt", "r") as file:
    #     for line in file.readlines():
    #         dictionary.append(line.strip())
    #
    # df = pd.DataFrame(tokens[:5], columns=dictionary)
    # df.to_csv("tokens.csv")

    positive_labels = np.ones(len(positive_lines))
    negative_labels = np.zeros(len(negative_lines))

    labels = np.append(positive_labels, negative_labels)

    return tokens, labels


def encode_data():
    with open("positive.txt", "r") as file:
        positive_lines = file.readlines()
    with open("negative.txt", "r") as file:
        negative_lines = file.readlines()

    dictionary = []
    with open("vocab.txt", "r") as file:
        for line in file.readlines():
            dictionary.append(line.strip())

    lines = np.append(positive_lines, negative_lines)
    encoded_data = []
    for line in lines:
        encoded_line = np.array([dictionary.index(word) for word in line.split()])
        encoded_line = np.add(encoded_line, 1)
        encoded_data.append(encoded_line)

    np.save("encoded_reviews.npy", np.array(encoded_data))


def load_encoded_data():
    encoded_data = np.load("encoded_reviews.npy", allow_pickle=True)
    return encoded_data