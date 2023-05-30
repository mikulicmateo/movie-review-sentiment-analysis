from sklearn import svm, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from data_util import load_data_into_CountVector, load_encoded_data
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras


def plot_learning_history(history, dropout, dim, lr, l2):
    txt = f"dropout: {dropout}, LSTM_dim: {dim}, Learning Rate: {lr}, l2_lambda: {l2}"

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.figtext(0.5, 0.001, txt, wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()



def remove_outliers(reviews_int, labels):

    # pd.Series(review_length).hist()
    # plt.show()
    # print(pd.Series(review_length).describe())
    # remove_outliers(reviews_int)

    review_length = [len(line) for line in reviews_int]
    q1_bound = pd.Series(review_length).quantile(0.25)
    q3_bound = pd.Series(review_length).quantile(0.75)
    iqr = q3_bound - q1_bound

    low_bound = q1_bound - 1.5 * iqr
    high_bound = q3_bound + 1.5 * iqr

    print(iqr)
    print((low_bound, high_bound))

    reviews_int = [reviews_int[i] for i, length in enumerate(review_length) if low_bound <= length <= high_bound]
    labels = [labels[i] for i, length in enumerate(review_length) if low_bound <= length <= high_bound]

    return np.array(reviews_int), np.array(labels)

def pad_features(reviews_int):
    #
    # Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    #

    review_length = [len(line) for line in reviews_int]
    seq_length = np.max(review_length)

    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = np.array(np.zeros(seq_length - review_len))
            new = np.concatenate((zeroes, review))
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i, :] = np.array(new)

    return features


def main():
    tokens, labels = load_data_into_CountVector()
    encoded_reviews, labels = remove_outliers(load_encoded_data(), labels)
    encoded_reviews = pad_features(encoded_reviews)
    vocab_size = 14803

    dropout = 0.5
    lstm_dim = 128
    lr = 3e-4
    l2_lambda = 0.1

    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, lstm_dim),
        tf.keras.layers.LSTM(lstm_dim,
                             kernel_initializer="he_normal",
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.optimizers.AdamW(learning_rate=lr)
    lstm_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    history = lstm_model.fit(encoded_reviews, labels, epochs=50, validation_split=0.3)

    plot_learning_history(history, dropout, lstm_dim, lr, l2_lambda)

    random_state = 85
    svm_model = svm.SVC(kernel='linear', random_state=random_state)
    nb_model = naive_bayes.BernoulliNB()
    me_model = LogisticRegression(random_state=random_state)

    #kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    #svm_scores = cross_val_score(svm_model, tokens, labels, cv=kfold)
    #nb_scores = cross_val_score(nb_model, tokens, labels, cv=kfold)
    #me_scores = cross_val_score(me_model, tokens, labels, cv=kfold)
#
    #print(f"SVM: ", svm_scores, " Avg: ", np.average(svm_scores))
    #print(f"NB: ", nb_scores, " Avg: ", np.average(nb_scores))
    #print(f"ME: ", me_scores, " Avg: ", np.average(me_scores))


if __name__ == "__main__":
    main()
