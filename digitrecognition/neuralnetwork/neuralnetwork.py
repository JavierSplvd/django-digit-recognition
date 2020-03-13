from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics   
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

class DigitNeuralNetwork:
    def __init__(self):
        self.digits = datasets.load_digits()

        X_train, X_test, y_train, y_test = train_test_split(
            self.digits['data'], self.digits['target'], test_size=0.3, shuffle=True)

        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(400), random_state=1)
        self.clf.fit(X_train, y_train)
        print(self.clf.score(X_test, y_test))
        predicted = self.clf.predict(X_test)
        disp = metrics.plot_confusion_matrix(self.clf, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        plt.show()

    def predict(self, x):
        return self.clf.predict(x)
        
    def predict_proba(self, x):
        return self.clf.predict_proba(x)

# d = DigitNeuralNetwork()
# print(datasets.load_digits()['target'])
# from PIL import Image
# import numpy
# a = d.digits['data'][0]
# im = Image.fromarray(numpy.asarray(a))
# im.show()