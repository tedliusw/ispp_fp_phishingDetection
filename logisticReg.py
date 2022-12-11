import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn import metrics


def main(): 
  print('Running Logistic Regression')
  ds = pd.read_csv("./dataset/dataset_phishing.csv")
  col_names = list(range(1, 88))
  X = ds.iloc[:, col_names]
  Y = ds.iloc[:, 88]
  Y = Y.replace("legitimate", 0)
  Y = Y.replace("phishing", 1)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle = True, test_size=0.2)
  lgc1 = LogisticRegression(solver="liblinear", max_iter=2500)
  lgc1.fit(X_train, y_train)
  y_pred = lgc1.predict(X_test)
  print(metrics.accuracy_score(y_test, y_pred))


  if len(sys.argv[1:]) > 0:
    websites = list(ds.iloc[:, 0])
    y_test, y_pred = list(y_test), list(y_pred)
    false_positive = []
    false_negative = []
    for i in range(len(y_pred)):
      if y_test[i] == 0 and y_pred[i] == 1:
        false_positive.append(websites[X_test.index[i]])
      if y_test[i] == 1 and y_pred[i] == 0:
        false_negative.append(websites[X_test.index[i]])
    
    print("False Positive: ", false_positive, len(false_positive)/len(y_test))
    print("False Negative: ", false_negative, len(false_negative)/len(y_test))

  lgc = LogisticRegression(solver="liblinear", max_iter=2500)
  cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=None)
  score = 0
  accuracy = 0
  for i in range(10):
    score += cross_val_score(lgc, X, Y, cv=cv, scoring='f1').mean()
    accuracy += cross_val_score(lgc, X, Y, cv=cv, scoring='accuracy').mean()
  score = score/10
  accuracy = accuracy/10

  print('Acu: ',accuracy)
  # print(accuracy.mean())
  print('f1: ',score)
  # print(scores.mean())



if __name__ == "__main__": 
  main()