from sklearn import datasets
from sklearn import svm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# load SVM
clf = svm.SVC()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random.randrange(100))

clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

target_names = ['setosa', 'versicolor', 'virginica']
print(classification_report(y_test, y_pred, target_names=target_names))
