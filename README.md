# pipeline_with_random_forest
The pipeline takes data from the database (SQLite), prepares the data, makes predictions using the trained random forest model, and saves the result of the predictions to a new table.

##  Prerequisites

This package requires:

- [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [Matplotlib](https://matplotlib.org/3.5.1/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)

## Description
1) ```modeling.py``` works with the initial data provided by the link [train_data_200k.csv](https://drive.google.com/file/d/1BGX6cpzauVE6tLAxfTjNYIVOKHU7CrtM/view?usp=sharing). Script creates and saves a model of Random Forest regression aimed to predict the values of the Target_1...4 parameters using the values of Tag_1...79.

2) ```pipeline.py```  takes data from the database (SQLite [test_data_100k](https://drive.google.com/file/d/12QEwrSQt1tCBbYwDCs-hDvMza47IyP2R/view?usp=sharing)), prepares the data, makes predictions using the trained model by ```modeling.py```, and saves the result of the predictions, top 10 features by importance to a new table file.

## Usage
Files train_data_200k.csv and test_data_100k should be within the working directory. To start the program, run:
```sh
pipeline.py
```
it imports ```modeling.py``` which stores the model for further prediction. If the file with the model has already been generated, ```pipeline.py``` will import it directly and make predictions.

## Note
The program might take several minutes to perform modeling.

## Author

👤 **Aleksei Krasnov**

* Website: [Ph.D. Aleksei Krasnov](https://www.researchgate.net/profile/Aleksei-Krasnov)
* Twitter: [@AlekseiKrasnov4](https://twitter.com/AlekseiKrasnov4)
* Github: [alexey-krasnov](https://github.com/alexey-krasnov)
* LinkedIn: [Aleksei Krasnov](https://linkedin.com/in/aleksei-krasnov-b53b2ab6)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/alexey-krasnov/random_forest_modeling/issues). 
