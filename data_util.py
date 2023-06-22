from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np


def load_data_into_count_vector():
    with open("Preprocessed_Data/Updated_Dataset/positive.txt", "r") as file:
        positive_lines = file.readlines()
    with open("Preprocessed_Data/Updated_Dataset/negative.txt", "r") as file:
        negative_lines = file.readlines()    

    lines = np.append(positive_lines, negative_lines)
    count_vec = CountVectorizer(binary=True)
    count_matrix = count_vec.fit_transform(lines)
    tokens = csr_matrix(count_matrix)

    positive_labels = np.ones(len(positive_lines))
    negative_labels = np.zeros(len(negative_lines))

    labels = np.append(positive_labels, negative_labels)

    return tokens, labels


def encode_data():
    with open("Preprocessed_Data/Updated_Dataset/positive.txt", "r") as file:
        positive_lines = file.readlines()
    with open("Preprocessed_Data/Updated_Dataset/negative.txt", "r") as file:
        negative_lines = file.readlines()

    dictionary = []
    with open("Preprocessed_Data/Updated_Dataset/vocab.txt", "r") as file:
        for line in file.readlines():
            dictionary.append(line.strip())

    lines = np.append(positive_lines, negative_lines)
    encoded_data = []
    for line in lines:
        encoded_line = np.array([dictionary.index(word) for word in line.split()])
        encoded_line = np.add(encoded_line, 1)
        encoded_data.append(encoded_line)

    np.save("Preprocessed_Data/Updated_Dataset/encoded_reviews.npy", np.array(encoded_data))


def load_encoded_data():
    encoded_data = np.load("Preprocessed_Data/Updated_Dataset/encoded_reviews.npy", allow_pickle=True)
    return encoded_data
