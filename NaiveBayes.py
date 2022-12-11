import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn import metrics

from sklearn.metrics import confusion_matrix

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

  conf_matrix=confusion_matrix(y_test, y_pred)
  fig, ax = plt.subplots(figsize=(7.5, 7.5))
  ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
  for i in range(conf_matrix.shape[0]):
      for j in range(conf_matrix.shape[1]):
          ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
  
  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix', fontsize=18)
  plt.show()

  gnbc = GaussianNB()
  cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=None)
  score = 0
  accuracy = 0
  for i in range(10):
    score += cross_val_score(gnbc, X, Y, cv=cv, scoring='f1').mean()
    accuracy += cross_val_score(gnbc, X, Y, cv=cv, scoring='accuracy').mean()
  score = score/10
  accuracy = accuracy/10

  print('Acu: ',accuracy)
  # print(accuracy.mean())
  print('f1: ',score)
  # print(scores.mean())




if __name__ == "__main__": 
  main()