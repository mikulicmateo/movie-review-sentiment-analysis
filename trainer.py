from sklearn import svm, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from data_util import load_data


def main():
    tokens, labels = load_data()
    random_state = 85
    svm_model = svm.SVC(kernel='linear', random_state=random_state)
    nb_model = naive_bayes.BernoulliNB()
    me_model = LogisticRegression(random_state=random_state)

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    svm_scores = cross_val_score(svm_model, tokens, labels, cv=kfold)
    nb_scores = cross_val_score(nb_model, tokens, labels, cv=kfold)
    me_scores = cross_val_score(me_model, tokens, labels, cv=kfold)

    print(f"SVM: ", svm_scores, " Avg: ", np.average(svm_scores))
    print(f"NB: ", nb_scores, " Avg: ", np.average(nb_scores))
    print(f"ME: ", me_scores, " Avg: ", np.average(me_scores))


if __name__ == "__main__":
	main()