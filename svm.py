import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


from sklearn import metrics

def main(): 
  print('Running SVM')
  ds = pd.read_csv("./dataset/dataset_phishing.csv")
  ds.head(0)

  col_names = list(range(1, 88))
  X = ds.iloc[:, col_names]
  Y = ds.iloc[:, 88]
  Y = Y.replace("legitimate", 0)
  Y = Y.replace("phishing", 1)
  print(Y)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle = True, test_size=0.2)
  clf1 = make_pipeline(StandardScaler(), SVC(kernel='sigmoid', gamma='auto'))
  clf1.fit(X_train, y_train)
  y_pred = clf1.predict(X_test)
  print(metrics.accuracy_score(y_test, y_pred))

  clf = make_pipeline(StandardScaler(), SVC(kernel='sigmoid', gamma='auto'))
  cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=None)
  score = 0
  accuracy = 0
  for i in range(10):
    score += cross_val_score(clf, X, Y, cv=cv, scoring='f1').mean()
    accuracy += cross_val_score(clf, X, Y, cv=cv, scoring='accuracy').mean()
  score = score/10
  accuracy = accuracy/10

  print('Acu: ',accuracy)
  # print(accuracy.mean())
  print('f1: ',score)
  # print(scores.mean())


if __name__ == "__main__":
  main()