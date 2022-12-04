import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

def main(): 
  print("Running Naive Bayes")
  ds = pd.read_csv("./dataset/dataset_phishing.csv")
  ds.head(0)

  col_names = list(range(1, 88))
  X = ds.iloc[:, col_names]
  Y = ds.iloc[:, 88]
  Y = Y.replace("legitimate", 0)
  Y = Y.replace("phishing", 1)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size = 0.2)
  gnb = GaussianNB()
  gnb.fit(X_train, y_train)

  y_pred = gnb.predict(X_test)
  print(metrics.accuracy_score(y_test, y_pred))



if __name__ == "__main__": 
  main()