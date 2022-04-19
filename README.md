• Using the initial data provided by the link Train_Data_200k, create a model aimed at predicting the values ​​of the Target_1...4 parameters using the values ​​of Tag_1...79.
• Apply the function to the test sample presented in the test_data_100k file and, based on the Tag_1...79 values, obtain predictions for the Target_1...4 parameters.
• After receiving the model, write a pipeline that takes data from the database (SQLite), prepares the data, makes predictions using the trained model, and saves the result of the predictions to a new table.
• Received forecasts, top 10 significant tags, workbook and pipeline, send as a result in a response letter.


1) modeling.py Script for fitting and saving ensemble model of Random Forest
 based on data from train_data_200k.csv. 

2) pipeline.py Pipeline that takes data from the database (SQLite),
prepares the data, makes predictions using the trained model by modeling.py,
and saves the result of the predictions, top 10 features by importance to a new tables.
