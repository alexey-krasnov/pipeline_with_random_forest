# -*- coding: UTF-8 -*-
"""Script for fitting and saving model of Random Forest regression
 based on the data from train_data_200k.csv"""


import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def open_train_dataset(data_name):
    """Open train dataset and fill NaN values with mean along column"""
    df = pd.read_csv(data_name)
    return df


def prepare_data(df):
    """Fill NaN values with mean along column"""
    for f in df.columns.values:
        df[f].fillna(df[f].mean(), inplace=True)
    return df


def get_features_names(df):
    """Get features names of DataFrame"""
    features_names = df.columns[:-4]
    return features_names


def make_x_y(df, features_names):
    """Transform features and target to Numpy arrays"""
    X = df[features_names].values
    y = df[['target1', 'target2', 'target3', 'target4']].values
    return X, y


def evaluate_model(X_train_stand, X_test_stand, y_train, y_test):
    """Estimation of model performance based on train_200k dataset"""
    rf_for_estimation = RandomForestRegressor(n_jobs=-1)
    rf_for_estimation.fit(X_train_stand, y_train)
    y_pred = rf_for_estimation.predict(X_test_stand)
    print("Random forest accuracy:", rf_for_estimation.score(X_test_stand, y_test))
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_test, y_pred, squared=False))
    print('Explained Variance Score:', metrics.explained_variance_score(y_test, y_pred))
    print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred))
    print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred))


def make_final_model(X_stand, y, model_name):
    """ Make final model, based on whole train_200k dataset"""
    rf_final = RandomForestRegressor(n_jobs=-1, random_state=101)
    rf_final.fit(X_stand, y)
    joblib.dump(rf_final, f'{model_name}', compress=3)
    return rf_final


def main(model_name):
    """Main function to perform training and make model of random forest """
    TRAIN_DATA = 'train_data_200k.csv'
    scaler = StandardScaler()
    train_df = open_train_dataset(data_name=TRAIN_DATA)
    train_df = prepare_data(df=train_df)
    features_names = get_features_names(df=train_df)
    X, y = make_x_y(train_df, features_names)
    # Split of Train Data 200k to evaluate model accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
    # Standardization of train/test subset features and all features
    X_train_stand = scaler.fit_transform(X_train)
    X_test_stand = scaler.fit_transform(X_test)
    X_stand = scaler.fit_transform(X)
    evaluate_model(X_train_stand, X_test_stand, y_train, y_test)
    return make_final_model(X_stand, y, model_name)


MODEL_NAME = 'rf_final_model.pkl'
if __name__ == "__main__":
    main(MODEL_NAME)
