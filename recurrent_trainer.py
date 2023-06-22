import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from data_util import load_data_into_count_vector, load_encoded_data

warnings.filterwarnings("ignore")

rnn_dim = 32
rnn_lr = 3e-4
rnn_dropout = 0.7
rnn_l2_lambda = 0.01
rnn_patience = 3

lstm_dim = 64
lstm_lr = 3e-4
lstm_dropout = 0.7
lstm_l2_lambda = 0.1
lstm_patience = 2

gru_dim = 100
gru_lr = 3e-4
gru_dropout = 0.5
gru_l2_lambda = 0.01
gru_patience = 1

train_splits = 3
vocab_size = 50183  # Old Vocab size: 14803
max_epoch = 15
random_state = 85


def plot_learning_history(history, model_name, fold, dropout, dim, lr, l2, removed_outliers):
    txt = f"dropout: {dropout}, nn_dim: {dim}, Learning Rate: {lr}, l2_lambda: {l2}"

    plt.plot(history.history['accuracy'])
    try:
        plt.plot(history.history['val_accuracy'])
    except:
        print("No val history")

    plt.title(f'{model_name} accuracy, removed outliers: {removed_outliers}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.figtext(0.5, 0.001, txt, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(os.path.join(os.getcwd(), f"plots/{model_name}_fold{fold}_rm_outliers_{removed_outliers}.png"))
    plt.show()


def remove_outliers(reviews_int, labels):
    review_length = [len(line) for line in reviews_int]

    pd.Series(review_length).hist()
    plt.title(f'Distribucija duljine recenzija')
    plt.ylabel('Količina recenzija')
    plt.xlabel('Broj riječi')
    plt.grid(None)
    plt.show()

    print(pd.Series(review_length).describe())

    q1_bound = pd.Series(review_length).quantile(0.25)
    q3_bound = pd.Series(review_length).quantile(0.75)
    iqr = q3_bound - q1_bound

    low_bound = q1_bound - 1.5 * iqr
    high_bound = q3_bound + 1.5 * iqr

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


def create_RNN(vocab_size, nn_dim, dropout, l2_lambda):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, nn_dim, mask_zero=True),
        tf.keras.layers.SimpleRNN(nn_dim, dropout=dropout, kernel_initializer="he_normal",
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_LSTM(vocab_size, nn_dim, dropout, l2_lambda):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, nn_dim, mask_zero=True),
        tf.keras.layers.LSTM(nn_dim, dropout=dropout, kernel_initializer="he_normal",
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_GRU(vocab_size, nn_dim, dropout, l2_lambda):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, nn_dim, mask_zero=True),
        tf.keras.layers.GRU(nn_dim, dropout=dropout, kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


def create_model(model_name, vocab_size, nn_dim, lr, dropout, l2_lambda):
    if model_name == "rnn":
        model = create_RNN(vocab_size, nn_dim, dropout, l2_lambda)
    elif model_name == "lstm":
        model = create_LSTM(vocab_size, nn_dim, dropout, l2_lambda)
    elif model_name == "gru":
        model = create_GRU(vocab_size, nn_dim, dropout, l2_lambda)
    else:
        print("ERROR: Please specify valid model")
        return None

    optimizer = tf.optimizers.AdamW(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


def kfold_train(encoded_reviews, labels, random_state, removed_outliers):
    accuracies_rnn = []
    accuracies_lstm = []
    accuracies_gru = []

    fold = 1
    for train_index, test_index in StratifiedKFold(n_splits=train_splits, shuffle=True,
                                                   random_state=random_state).split(
            encoded_reviews, labels):
        x_train, x_test = encoded_reviews[train_index], encoded_reviews[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        print(f"---------- TRAINING RNN FOLD {fold}/{train_splits} ----------")
        early_stop_rnn = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=rnn_patience,
                                                          start_from_epoch=8)
        model_rnn = create_model("rnn", vocab_size, rnn_dim, rnn_lr, rnn_dropout, rnn_l2_lambda)
        train_history_rnn = model_rnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epoch,
                                          callbacks=[early_stop_rnn])

        accuracies_rnn.append(np.max(train_history_rnn.history['val_accuracy']))
        plot_learning_history(train_history_rnn, "rnn", fold, rnn_dropout, rnn_dim, rnn_lr, rnn_l2_lambda,
                              removed_outliers)

        print(f"---------- TRAINING LSTM FOLD {fold}/{train_splits} ----------")
        model_lstm = create_model("lstm", vocab_size, lstm_dim, lstm_lr, lstm_dropout, lstm_l2_lambda)
        early_stop_lstm = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=lstm_patience,
                                                           start_from_epoch=8)
        train_history_lstm = model_lstm.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epoch,
                                            callbacks=[early_stop_lstm])
        accuracies_lstm.append(np.max(train_history_lstm.history['val_accuracy']))
        plot_learning_history(train_history_lstm, "lstm", fold, lstm_dropout, lstm_dim, lstm_lr, lstm_l2_lambda,
                              removed_outliers)

        print(f"---------- TRAINING GRU FOLD {fold}/{train_splits}----------")
        model_gru = create_model("gru", vocab_size, gru_dim, gru_lr, gru_dropout, gru_l2_lambda)
        early_stop_gru = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=gru_patience,
                                                          start_from_epoch=8)
        train_history_gru = model_gru.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epoch,
                                          callbacks=[early_stop_gru])
        accuracies_gru.append(np.max(train_history_gru.history['val_accuracy']))
        plot_learning_history(train_history_gru, "gru", fold, gru_dropout, gru_dim, gru_lr, gru_l2_lambda,
                              removed_outliers)

        fold += 1

    print("Average accuracy RNN: ", np.average(accuracies_rnn))
    print("Average accuracy LSTM: ", np.average(accuracies_lstm))
    print("Average accuracy GRU: ", np.average(accuracies_gru))


def main():
    _, labels = load_data_into_count_vector()

    # without removing outliers
    removed_outliers = False
    encoded_reviews = pad_features(load_encoded_data())
    kfold_train(encoded_reviews, labels, random_state, removed_outliers)

    # removed outliers
    removed_outliers = True
    encoded_reviews, labels = remove_outliers(load_encoded_data(), labels)
    encoded_reviews = pad_features(encoded_reviews)
    kfold_train(encoded_reviews, labels, random_state, removed_outliers)


if __name__ == "__main__":
    main()
