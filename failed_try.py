import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('train.csv')
X = data.drop('y', axis=1)
y = data['y']
ridge_model = RidgeCV(alphas=[0.1, 10, 100, 200], cv=10)
scores = cross_val_score(ridge_model, X, y, cv = 10, scoring='neg_root_mean_squared_error')
print("Cross-validation scores:", scores)