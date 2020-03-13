import numpy as np
from sklearn.svm import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        self.digits = datasets.load_digits()

        X_train, X_test, y_train, y_test = train_test_split(
            self.digits['data'], self.digits['target'], test_size=0.3, shuffle=True)
        self.clf = LinearSVC(dual=False, C=1, verbose=1)
        self.clf.fit(X_train, y_train)

        print(self.clf.score(X_test, y_test))

        # PLotting
        disp = metrics.plot_confusion_matrix(self.clf, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        plt.show()

svm = SVM()