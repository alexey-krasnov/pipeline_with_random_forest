# -*- coding: UTF-8 -*-
"""Pipeline that takes data from the database (SQLite),
prepares the data, makes predictions using the trained model by modeling.py,
and saves: the results of the predictions, top 10 features by importance to the new table files"""


import sqlite3
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import modeling


def database_reading(*args):
    """ Return DataFrame from database."""
    if not args:
        # Direct reading from test_data_100k.csv file to test program.
        df = pd.read_csv('test_data_100k.csv', index_col=False)
        return df
    database_name, table_name = args
    conn = sqlite3.connect(database_name)
    # Open test_100k as table in database.db
    curs = conn.cursor()
    query = f'SELECT * FROM {table_name}'
    # # Create Pandas DataFrame based on table from database
    df = pd.read_sql_query(query, conn).fillna(0).values
    curs.close()
    conn.close()
    return df


def prepare_data_frame(df):
    """Fill NaN values with mean along column"""
    for f in df.columns.values:
        df[f].fillna(df[f].mean(), inplace=True)
    return df


def get_features_names(df):
    """Get features names of DataFrame"""
    features_names = df.columns
    return features_names


def make_x(df, features_names):
    """Transform features to Numpy array"""
    X = df[features_names].values
    return X


def feats_standardization(X):
    """Standardization of features"""
    scaler = StandardScaler()
    X_stand = scaler.fit_transform(X)
    return X_stand


def load_model(model_name):
    """Load fitted final model of Random Forest"""
    try:
        rf_final_loaded = joblib.load(model_name)
    except FileNotFoundError:
        rf_final_loaded = modeling.main(model_name)
    return rf_final_loaded


def predict_model(model, X_stand):
    """Make prediction based on loaded model"""
    y_pred = model.predict(X_stand)
    predicted_data_frame = pd.DataFrame(y_pred, columns=['target1', 'target2',
                                                         'target3', 'target4'])
    return predicted_data_frame

def hist_plot(predict_data_frame):
    """ Plot histogram of predicted values"""
    predict_data_frame.hist()
    plt.savefig('target_predicted.png', dpi=300)


def top_10_feats(model, features_names):
    """Export top 10 Features by Importance as features_by_importance.csv file"""
    feats_imp = pd.Series(model.feature_importances_,
                          index=features_names).sort_values(ascending=False)
    feats_imp[:10].to_csv('features_by_importance.csv', header=['Feature Importance'])


if __name__ == "__main__":
    loaded_df = database_reading()

    working_df = prepare_data_frame(loaded_df)
    features = get_features_names(df=working_df)

    X = make_x(df=working_df, features_names=features)
    X_stand = feats_standardization(X)

    loaded_model = load_model(modeling.MODEL_NAME)
    prediction = predict_model(loaded_model, X_stand)

    # Export predicted values as prediction.csv file
    prediction.to_csv('prediction.csv', index=False)
    # prediction.to_parquet('prediction.parquet', index=False, engine='fastparquet')

    hist_plot(prediction)
    top_10_feats(loaded_model, features)
