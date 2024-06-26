import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

def main():
    """
    Main workflow function for training the model, making predictions, 
    and plotting important features.
    """
    # Read the data
    train_data = pd.read_csv('data_train.csv')
    test_data = pd.read_csv('data_test.csv')

    # Separate features and target
    X = train_data.drop(['patient_id', 'patient_gender', 'treatment_pd'], axis=1)
    y = train_data['treatment_pd']

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a combined DataFrame for AutoGluon
    train_dataset = X_train.copy()
    train_dataset['treatment_pd'] = y_train

    # Set up AutoGluon
    time_limit = 3600 * 10
    automl = TabularPredictor(label='treatment_pd', problem_type='regression', eval_metric='root_mean_squared_error')

    # Train the model
    automl.fit(train_dataset, presets='medium_quality', time_limit=time_limit, num_bag_folds=5, 
               num_bag_sets=0, num_stack_levels=0, dynamic_stacking=False, 
               included_model_types=['XGB', 'CAT', 'GBM'], ag_args_fit={'num_cpus': 4})

    # Prepare the test set for predictions
    test_dataset = X_test.copy()
    test_dataset['treatment_pd'] = y_test

    # Predictions on the test set
    predictions = automl.predict(test_dataset.drop(columns=['treatment_pd']))
    print("Predictions completed")

    # Feature selection using feature importance from AutoGluon
    feature_importance = automl.feature_importance(train_dataset)
    print("Feature importance:\n", feature_importance)

    # Select top features with importance greater than 0.3
    top_features = feature_importance[feature_importance['importance'] > 0.3].index.tolist()
    print("Top important features:\n", top_features)


    # Return the results
    return predictions, feature_importance