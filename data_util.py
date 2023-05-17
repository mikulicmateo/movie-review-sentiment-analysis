from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def load_data():
    with open("positive.txt", "r") as file:
        positive_lines = file.readlines()
    with open("negative.txt", "r") as file:
        negative_lines = file.readlines()    

    lines = np.append(positive_lines, negative_lines)
    count_vec = CountVectorizer(stop_words='english', binary=True)
    count_matrix = count_vec.fit_transform(lines)
    tokens = count_matrix.toarray()

    positive_labels = np.ones(len(positive_lines))
    negative_labels = np.zeros(len(negative_lines))

    labels = np.append(positive_labels, negative_labels)

    return tokens, labels