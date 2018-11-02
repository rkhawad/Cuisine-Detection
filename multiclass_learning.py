from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier

iris = datasets.load_iris()

print iris
X, y = iris.data, iris.target
clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)

print X
print y
print clf.fit(X, y).predict(X)

