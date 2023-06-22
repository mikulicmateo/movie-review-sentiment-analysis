from sklearn import svm, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

from data_util import load_data_into_count_vector


def main():
    print("Loading Data")

    tokens, labels = load_data_into_count_vector()

    print("Data Loaded")

    random_state = 85
    svm_model = svm.SVC(kernel='linear', random_state=random_state)
    nb_model = naive_bayes.BernoulliNB()
    me_model = LogisticRegression(random_state=random_state)

    print("Making K-fold")
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    print("K-fold made")

    print("Training Started")
    svm_scores = cross_val_score(svm_model, tokens, labels, cv=kfold)
    print(f"SVM: ", svm_scores, " Avg: ", np.average(svm_scores))

    nb_scores = cross_val_score(nb_model, tokens, labels, cv=kfold)
    print(f"NB: ", nb_scores, " Avg: ", np.average(nb_scores))

    me_scores = cross_val_score(me_model, tokens, labels, cv=kfold)
    print(f"ME: ", me_scores, " Avg: ", np.average(me_scores))


if __name__ == "__main__":
    main()
