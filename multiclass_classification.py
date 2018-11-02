import numpy as np
import sys
import json
import random
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def partialFit(model, X, Y, binarizer, n_iterations, kernel='linear', n_components=100, gamma=1.0):
	print kernel
	shuffledRange = range(len(X))
	for iterations in range(n_iterations):
		random.shuffle(shuffledRange)
		shuffledX = [X[i] for i in shuffledRange]
		shuffledY = [Y[i] for i in shuffledRange]
		for batch in batches(range(len(shuffledX)), 4000):
			X_batch = binarizer.transform(shuffledX[batch[0]:batch[-1] + 1])
			Y_batch = shuffledY[batch[0]:batch[-1] + 1]
			if kernel == 'rbf':
				rbf_feature = RBFSampler(gamma=0.5, random_state=1, n_components=n_components)
				X_batch = rbf_feature.fit_transform(X_batch)
			model.partial_fit(X_batch, Y_batch, classes=np.unique(Y))

def parseData():
	f1 = open(sys.argv[1], 'r')
	f2 = open(sys.argv[2], 'r')

	cuisine_list = map(lambda x : x.strip(), f1.readlines())
	ingredients_list = map(lambda x: x.strip(), f2.readlines())

	f1.close()
	f2.close()

	cuisine_indices = [i for i in range(len(cuisine_list))]
	ingredients_indices = [i for i in range(len(ingredients_list))]


	cuisine_map = dict(zip(cuisine_list, cuisine_indices))
	ingredients_map = dict(zip(ingredients_list, ingredients_indices))

	cuisine_indices = []
	ingredients_indices = []

	with open(sys.argv[3]) as data_file:    
	  data = json.load(data_file)

	Y = []
	X = []
	for data_set in data:
	  ingredients = []
	  Y.append(cuisine_map[data_set["cuisine"]])
	  for ingredient in data_set["ingredients"]:
	    ingredients.append(ingredients_map[ingredient])
	  X.append(ingredients)

	X = np.array(X)
	Y = np.array(Y)
	binarizer = MultiLabelBinarizer()
	binarizer.fit([range(len(ingredients_list))])
	return X, Y, binarizer

def buildData(X, Y, test_ratio):

	split = int((1 - (test_ratio * 1.0) / 100) * len(Y))

	X_train = X[:split]
	Y_train = Y[:split]
	X_test = X[split:]
	Y_test = Y[split:]
	return X_train, Y_train, X_test, Y_test 

def checkAccuracy(clf, binarizer, features, labels, kernel='kernel', n_components=100, gamma=1.0):
	features = binarizer.transform(features)
	if kernel == 'rbf':
		rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components=n_components)
		features = rbf_feature.fit_transform(features)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	print pred
	return accuracy

def trainKernelBasedSVM(test_ratio):
	X, Y, binarizer = parseData()
	X_train, Y_train, X_test, Y_test = buildData(X, Y, test_ratio)
	gammas_list = [0.5, 0.2, 0.4, 0.1, 0.01]
	for gamma in gammas_list:
		clf = linear_model.SGDClassifier(n_iter=10)
		partialFit(clf, X_train, Y_train, binarizer, 10, 'rbf', 4598, gamma)
		print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test, 'rbf', 4598, gamma))

def trainLinearSVM(test_ratio):
	X, Y, binarizer = parseData()
	X_train, Y_train, X_test, Y_test = buildData(X, Y, test_ratio)
	clf = linear_model.SGDClassifier(n_iter=10)
	partialFit(clf, X_train, Y_train, binarizer, 10)
	print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test))

def trainMLPRegressor(test_ratio):
	X, Y, binarizer = parseData()
	X_train, Y_train, X_test, Y_test =  buildData(X, Y, test_ratio)
	for momentum in [0, .9]:
		clf = MLPRegressor(solver='sgd', activation='relu', learning_rate_init=0.01, random_state=1, momentum=momentum)
		partialFit(clf, X_train, Y_train, binarizer, 10)
		print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test))

def trainMLPClassifier(test_ratio):
	X, Y, binarizer = parseData()
	X_train, Y_train, X_test, Y_test = buildData(X, Y, test_ratio)
	clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5000, 2), random_state=1, learning_rate='adaptive', max_iter=400)
	partialFit(clf, X_train, Y_train, binarizer, 10)
	print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test))

def naiveBayes(test_ratio):
	X, Y, binarizer = parseData()
	X_train, Y_train, X_test, Y_test = buildData(X, Y, test_ratio)
	clf = GaussianNB()
	partialFit(clf, X_train, Y_train, binarizer, 10)
	print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test))
	clf = MultinomialNB()
	partialFit(clf, X_train, Y_train, binarizer, 10)
	print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test))
	clf = BernoulliNB()
	partialFit(clf, X_train, Y_train, binarizer, 10)
	print "Accuracy " + str(checkAccuracy(clf, binarizer, X_test, Y_test))

if __name__ == '__main__':

	naiveBayes(20)
	trainKernelBasedSVM(20)
	trainLinearSVM(20)
	trainMLPClassifier(20)
	trainMLPRegressor(20)
	trainKernelBasedSVM(30)
	trainKernelBasedSVM(20)
	trainANN(20)
	#trainANN(10)