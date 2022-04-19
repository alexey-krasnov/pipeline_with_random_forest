##  Prerequisites

This package requires:

- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [Matplotlib](https://matplotlib.org/3.5.1/)
- [Sklearn](https://scikit-learn.org/stable/)
- [joblib](https://joblib.readthedocs.io/en/latest/)

## Usage
1) modeling.py Script for fitting and saving model of Random Forest regression based on data from [train_data_200k.csv](https://drive.google.com/file/d/1RQPXq6cBFBRNOJD1-5QGBtA2tuNbQvZr/view?usp=sharing). 

2) pipeline.py Pipeline that takes data from the database (SQLite [test_data_100k](https://drive.google.com/file/d/15hEL073pA1Vag74nwig_fcPHN-44f1JP/view?usp=sharing)), prepares the data, makes predictions using the trained model by modeling.py, and saves the result of the predictions, top 10 features by importance to a new table files.

## Author

👤 **Aleksei Krasnov**

* Website: https://www.researchgate.net/profile/Aleksei-Krasnov
* Twitter: [@AlekseiKrasnov4](https://twitter.com/AlekseiKrasnov4)
* Github: [@alexey-krasnov](https://github.com/alexey-krasnov)
* LinkedIn: [@aleksei-krasnov-b53b2ab6](https://linkedin.com/in/aleksei-krasnov-b53b2ab6)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/alexey-krasnov/absorption_tauc_plot/issues). 
