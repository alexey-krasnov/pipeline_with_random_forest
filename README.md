1) modeling.py Script for fitting and saving ensemble model of Random Forest
 based on data from train_data_200k.csv. 

2) pipeline.py Pipeline that takes data from the database (SQLite),
prepares the data, makes predictions using the trained model by modeling.py,
and saves the result of the predictions, top 10 features by importance to a new tables.
