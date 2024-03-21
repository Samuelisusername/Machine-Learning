
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from  sklearn.linear_model._ridge import Ridge
from sklearn.model_selection import cross_val_score




def average_LR_RMSE(X, y, lambdas, n_folds):
    
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    for i in range(len(lambdas)):
        model = Ridge(alpha=lambdas[i], fit_intercept=False)
        RMSE_mat[:,i] = -1 * cross_val_score(model, X, y, cv=n_folds, scoring='neg_root_mean_squared_error')
        print("this is the lamda ", lambdas[i], np.average(RMSE_mat[:,i]))

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    print(data.head())

    X = data.to_numpy()
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")
