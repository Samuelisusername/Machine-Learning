# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
from statistics import LinearRegression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# Add any additional imports here (however, the task is solvable without using
# any additional imports)
# import ...

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here
    for j in range(700):
        for i in range(5):
            X_transformed[j][i] = X[j][i]
        for i in range(5):
            X_transformed[j][5 + i] = X[j][i] * X[j][i]
        for i in range(5):
            X_transformed[j][i + 10] = np.exp(X[j][i])
        for i in range(5):
            X_transformed[j][i + 15] = np.cos(X[j][i])
        X_transformed[j][20] = 1

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """

    w = np.zeros((21,))
    X_transformed = transform_data(X)
    weights = []
    min_mean_score = 1000000
    errors = []

    best_index = -1
    # TODO: Enter your code here
    lamdas = np.linspace(42.83321, 42.8334, num=10000)

    for i in range(len(lamdas)):
        model = Ridge(alpha=lamdas[i], fit_intercept = False)
        model.fit(X_transformed, y)
        weights = model.coef_
        scores = -1 * cross_val_score(model, X_transformed, y, cv=10, scoring='neg_root_mean_squared_error')
       # print(weights, "<- this are the weights")
        #print(scores, "<- this are the scores")
       # print(np.average(scores), lamdas[i])
        errors.append(np.average(scores))
        if(np.average(scores) < min_mean_score):
            min_mean_score = np.average(scores)
            best_index = i

    model = Ridge(alpha=lamdas[best_index], fit_intercept=False)
    model.fit(X_transformed, y)
    weights = model.coef_

    fig, ax = plt.subplots()
    ax.plot(lamdas, errors, linestyle="None", marker="o")
    plt.savefig("./plot1.png")

    scores = -1 * cross_val_score(model, X_transformed, y, cv=10, scoring='neg_root_mean_squared_error')

    assert w.shape == (21,)
    w = weights
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
