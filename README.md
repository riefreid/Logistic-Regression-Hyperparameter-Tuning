# Logistic Regression Hyperparameter Tuning

This repository contains Python code for performing hyperparameter tuning on a Logistic Regression model using different configurations. The dataset is loaded, preprocessed, and the model is trained and evaluated with various hyperparameter combinations.

## Dataset

The dataset is loaded from 'dataset.csv', assuming the 'label' column contains class labels. Columns 'Barcode' and 'label' are dropped, and the data is normalized using StandardScaler.

## Model Training

The dataset is split into training and test sets, and a Logistic Regression model is trained with different hyperparameter combinations. Hyperparameters include multi_class options ('ovr', 'multinomial') and penalty options ('l1', 'l2', 'elasticnet').

## Hyperparameter Tuning

Results for different hyperparameter combinations are stored, and a bar plot is generated to visualize accuracy for each configuration.

## Additional Tuning for Elastic Net Penalty

Further hyperparameter tuning is performed specifically for the elasticnet penalty, with a range of L1 ratios. The best performing L1 ratio is identified for both 'ovr' and 'multinomial' multi-class options.

## Usage

1. Clone the repository:
   git clone https://github.com/your-username/Logistic-Regression-Hyperparameter-Tuning.git
2. Run the Python script:
   python logistic_regression_hyperparameter_tuning.py
3. Explore the hyperparameter tuning results and identify the best configurations.

## Dependencies
- pandas
- scikit-learn
- matplotlib
