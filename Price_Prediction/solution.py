import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor

def data_loading():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Drop all columns where the CHF price field is empty
    train_df = train_df[train_df['price_CHF'].notna()]
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Initialize the train and test data to NaN
    X_train = np.empty_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.empty_like(train_df['price_CHF'])
    X_test = np.empty_like(test_df)
    X_train[:] = np.nan
    y_train[:] = np.nan
    X_test[:] = np.nan

    # Copy the data
    X_train = train_df.drop(['price_CHF'],axis=1)
    y_train = train_df['price_CHF']
    X_test = test_df

    # Transform string features into numeric features
    X_train.replace(to_replace='spring', value=1, inplace=True)
    X_train.replace(to_replace='summer', value=2, inplace=True)
    X_train.replace(to_replace='autumn', value=3, inplace=True)
    X_train.replace(to_replace='winter', value=4, inplace=True)
    X_test.replace(to_replace='spring', value=1, inplace=True)
    X_test.replace(to_replace='summer', value=2, inplace=True)
    X_test.replace(to_replace='autumn', value=3, inplace=True)
    X_test.replace(to_replace='winter', value=4, inplace=True)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    # Fill in the missing data based on other features
    imputer = IterativeImputer(max_iter=100, random_state=67)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.fit_transform(X_test)

    # Shuffle the data
    X_train_imputed, y_train = shuffle(X_train_imputed,y_train, random_state=0)

    # Train and predict
    kernel = RationalQuadratic(length_scale=0.5, alpha=1e4)
    model = GaussianProcessRegressor(kernel=kernel)
    scores = cross_val_score(model, X_train_imputed, y_train, cv=5, scoring='r2')
    print("scores=", scores)
    print("avg score=", np.average(scores), "for model=", model)
    model.fit(X_train_imputed, y_train)
    y_pred=np.zeros(X_test.shape[0])
    y_pred = model.predict(X_test_imputed)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

if __name__ == "__main__":
    X_train, y_train, X_test = data_loading()
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
