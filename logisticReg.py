import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn import metrics


def main(): 
  print('Running Logistic Regression')
  ds = pd.read_csv("./dataset/dataset_phishing.csv")
  ds.head(0)

  col_names = list(range(1, 88))
  X = ds.iloc[:, col_names]
  Y = ds.iloc[:, 88]
  Y = Y.replace("legitimate", 0)
  Y = Y.replace("phishing", 1)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle = True, test_size=0.2)
  lgc1 = LogisticRegression(solver="liblinear")
  lgc1.fit(X_train, y_train)
  y_pred = lgc1.predict(X_test)
  print(metrics.accuracy_score(y_test, y_pred))

if __name__ == "__main__": 
  main()