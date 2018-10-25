import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    data = pd.read_csv('someFile')
    (m, n) = data.shape
    feature_cols = ["List with" , "column titles"]
    X = data[feature_cols]
    Y = data["Y Column"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
