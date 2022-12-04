import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import plot_confusion_matrix

from sklearn import metrics

def main():
  print("Running Random Forest")
  ds = pd.read_csv("./dataset/dataset_phishing.csv")
  ds.head(0)

  col_names = list(range(1, 88))
  X = ds.iloc[:, col_names]
  Y = ds.iloc[:, 88]
  Y = Y.replace("legitimate", 0)
  Y = Y.replace("phishing", 1)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle = True, test_size=0.2)
  rfc1 = RandomForestClassifier(n_estimators=900, 
                             max_depth=20,
                             random_state=42)
  rfc1.fit(X_train, y_train)
  y_pred = rfc1.predict(X_test)
  print(metrics.accuracy_score(y_test, y_pred))

  rfc = RandomForestClassifier(n_estimators=900, 
                             max_depth=20,
                             random_state=42)
  cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=None)
  score = 0
  accuracy = 0
  for i in range(10):
    score += cross_val_score(rfc, X, Y, cv=cv, scoring='f1').mean()
    accuracy += cross_val_score(rfc, X, Y, cv=cv, scoring='accuracy').mean()
  score = score/10
  accuracy = accuracy/10

  print('Acu: ',accuracy)
  # print(accuracy.mean())
  print('f1: ',score)
  # print(scores.mean())


if __name__ == "__main__":
  main()