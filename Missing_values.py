import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from collections import Counter

# Exclude variables with high percent of missing values from data sets
def exclude_variables_with_high_missing_percentage(data, missing_percentages, threshold):
    """
    Exclude variables from the DataFrame with a higher percentage of missing data than the specified threshold.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - missing_percentages (Series): A Series containing the percentage of missing values for each column.
    - threshold (float): The threshold percentage above which variables will be excluded.

    Returns:
    - data_cleaned (DataFrame): The DataFrame with variables excluded based on the threshold.
    """
    # Exclude variables with a higher percentage of missing data than the specified threshold
    excluded_variables = [variable for variable, percentage in missing_percentages.items() if percentage > threshold]
    
    # Exclude the variables from the dataset
    data_cleaned = data.drop(excluded_variables, axis=1)
    
    return data_cleaned

# Display variables that have more than 50% missing data in the train and test set
def get_missing_percentages(data):
    """
    Calculate the percentage of missing values for columns with over 60% missing values.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.

    Returns:
    - missing_percentages (Series): A Series containing the percentage of missing values for columns with over 60% missing values.
    """
    nan_over_50_percent = [k for k, v in dict(data.isna().mean()).items() if v >= 0.6]
    missing_percentages = data[nan_over_50_percent].isna().mean().sort_values(ascending=False) * 100
    return missing_percentages


#Defining a function for missing values
def filter_na_cols(df):
    """
    Filter columns in the DataFrame that contain missing values.

    Parameters:
    - df (DataFrame): The DataFrame to filter.

    Returns:
    - result (Series or str): A Series containing the count of missing values for each column with missing values, or 'Clean dataset' if no missing values are found.
    """
    count_na_df = df.isna().sum()
    if count_na_df[count_na_df > 0].tolist():
        return count_na_df[count_na_df > 0]
    else:
        return 'Clean dataset'
    
    
# fill missing races
def fill_missing_race(df):
    """
    Fill missing values in the 'patient_race' column of the DataFrame based on the maximum percentage race in the region.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the 'patient_race' column to be filled.

    Returns:
    None
    """
    # Identify rows with missing values in patient_race
    missing_races = df.loc[pd.isna(df['patient_race']), 'patient_id'].unique()
    
    # Fill in the missing value in patient_race for each missing patient_race
    for target_id in missing_races:
        # Select the row corresponding to the patient_id
        target_row = df.loc[df['patient_id'] == target_id]

        # Check if there are missing values in patient_race
        if target_row['patient_race'].isnull().values.any():
            # Find the race with the maximum percentage in the region
            max_race = target_row[['race_white', 'race_black', 'race_asian', 'race_native', 'race_pacific', 'race_other', 'race_multiple']].idxmax(axis=1).values[0]

            # Replace the missing value in patient_race with the race corresponding to the maximum percentage
            df.loc[df['patient_id'] == target_id, 'patient_race'] = max_race.split('_')[-1].capitalize()

#fill missing values as vlookup function
def fill_missing_values(df, target_column, reference_column):
    """
    Fill missing values in the target column of the DataFrame using corresponding values from the reference column.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the target and reference columns.
    - target_column (str): The column with missing values to be filled.
    - reference_column (str): The column containing corresponding values used for filling missing values.

    Returns:
    None
    """
    # Find unique missing values in the target column
    missing_values = df.loc[pd.isna(df[target_column]), reference_column].unique()
    
    # Iterate through the missing values
    for target_value in missing_values:
        # Get the corresponding value from the reference column
        target_reference_series = df.loc[df[reference_column] == target_value, target_column].dropna()
        
        # Check if the series is not empty before accessing the first element
        if not target_reference_series.empty:
            target_reference = target_reference_series.iloc[0]
            
            # Replace missing values in the target column with the corresponding value from the reference column
            df.loc[df[reference_column] == target_value, target_column] = target_reference

#check for missing values and impute them
def check_and_fill_all_missing_values(df, first_reference_column, second_reference_column, exclude_columns=[]):
    """
    Check for missing values in the DataFrame and fill them using reference columns.
    
    Parameters:
    - df (DataFrame): The DataFrame to check and fill missing values.
    - first_reference_column (str): The column to use as the first reference for filling missing values.
    - second_reference_column (str): The column to use as the second reference if missing values remain after the first reference.
    - exclude_columns (list): A list of columns to exclude from the check and filling process.

    Returns:
    - df (DataFrame): The DataFrame with missing values filled using reference columns.
    """
    # Get columns with missing values, excluding specified columns
    missing_columns = [col for col in df.columns[df.isnull().any()] if col not in exclude_columns]

    for target_column in missing_columns:
        # Fill missing values using the first reference column
        fill_missing_values(df, target_column, first_reference_column)
        
        # Check again for missing values in the target column
        if df[target_column].isna().sum() > 0:
            # Fill missing values using the second reference column
            fill_missing_values(df, target_column, second_reference_column)
    
    return df

#define function for averaging imputation:
def mean_imputation(df1, df2, columns):
    """
    Perform mean imputation on missing values in specified columns of df1
    using the mean values calculated from corresponding columns in df2.

    Parameters:
    - df1 (DataFrame): The DataFrame containing missing values to be imputed.
    - df2 (DataFrame): The DataFrame used for calculating mean values.
    - columns (list): A list of column names to perform mean imputation on.

    Returns:
    - df1 (DataFrame): The DataFrame with missing values imputed using mean values.
    """
    for column in columns:
        mean_value = df2[column].mean()
        df1[column].fillna(mean_value, inplace=True)
    return df1

#define function for median imputation:
def median_imputation(df1, df2, columns):
    """
    Perform median imputation on missing values in specified columns of df1
    using the median values calculated from corresponding columns in df2.

    Parameters:
    - df1 (DataFrame): The DataFrame containing missing values to be imputed.
    - df2 (DataFrame): The DataFrame used for calculating median values.
    - columns (list): A list of column names to perform median imputation on.

    Returns:
    - df1 (DataFrame): The DataFrame with missing values imputed using median values.
    """
    for column in columns:
        median_value = df2[column].median()
        df1[column].fillna(median_value, inplace=True)
    return df1

#define function for mode imputation:
def mode_imputation(df1, df2, columns):
    """
    Perform mode imputation on missing values in specified columns of df1
    using the mode values calculated from corresponding columns in df2.

    Parameters:
    - df1 (DataFrame): The DataFrame containing missing values to be imputed.
    - df2 (DataFrame): The DataFrame used for calculating mode values.
    - columns (list): A list of column names to perform mode imputation on.

    Returns:
    - df1 (DataFrame): The DataFrame with missing values imputed using mode values.
    """
    for column in columns:
        mode_value = df2[column].mode().iloc[0]
        df1[column].fillna(mode_value, inplace=True)
    return df1 

# Function for mean imputation by group
def mean_imputation_by_column(df1, df2, columns, group_by_column):
    """
    Impute missing values in specified columns of DataFrame df1 based on mean values from DataFrame df2,
    grouped by a specified column.

    Args:
        df1 (pd.DataFrame): DataFrame containing missing values to be imputed.
        df2 (pd.DataFrame): DataFrame used as reference for imputation, containing the non-missing values.
        columns (list): List of column names in df1 for which missing values are to be imputed.
        group_by_column (str): Name of the column in df2 used for grouping.

    Returns:
        pd.DataFrame: DataFrame df1 with missing values imputed using mean values from df2.
    """
    for column in columns:
        for index, row in df1[df1[column].isnull()].iterrows():
            group_value = row[group_by_column]
            # Calculate the mean value for the group in df2
            mean_value = df2[df2[group_by_column] == group_value][column].mean()
            # Fill the missing value in df1 with the mean value
            df1.at[index, column] = mean_value
    return df1

# Function for median imputation by group
def median_imputation_by_column(df1, df2, columns, group_by_column):
    """
    Impute missing values in specified columns of DataFrame df1 based on median values from DataFrame df2,
    grouped by a specified column.

    Args:
        df1 (pd.DataFrame): DataFrame containing missing values to be imputed.
        df2 (pd.DataFrame): DataFrame used as reference for imputation, containing the non-missing values.
        columns (list): List of column names in df1 for which missing values are to be imputed.
        group_by_column (str): Name of the column in df2 used for grouping.

    Returns:
        pd.DataFrame: DataFrame df1 with missing values imputed using median values from df2.
    """
    for column in columns:
        for index, row in df1[df1[column].isnull()].iterrows():
            group_value = row[group_by_column]
            # Calculate the median value for the group in df2
            median_value = df2[df2[group_by_column] == group_value][column].median()
            # Fill the missing value in df1 with the median value
            df1.at[index, column] = median_value
    return df1

#replace missing values with logistic regression
def replace_missing_with_logistic_regression(df, target_column):
    """
    Replace missing values in a target column using logistic regression predictions.

    Args:
        df (pd.DataFrame): The DataFrame containing the target column with missing values.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: The DataFrame with missing values in the target column replaced.

    Note:
        This function uses logistic regression to predict missing values in the target column.
        It encodes categorical columns and trains a logistic regression model to predict the missing values.
        Missing values are replaced with the predicted values obtained from the logistic regression model.
    """
    # Check for missing values in target_column
    missing_values = df[target_column].isna()

    # Drop target_column and encode categorical columns
    Xdf = df.drop([target_column], axis=1)
    cate_data = Xdf.select_dtypes("object").columns
    Xdf = pd.get_dummies(Xdf, columns=cate_data)

    # Replace NaN with 'Unknown' in target_column
    NaN = np.nan
    df[target_column].replace(NaN, 'Unknown', inplace=True)

    # Prepare X and y for logistic regression
    ydf = df[target_column]

    # Train logistic regression model
    LR = LogisticRegression(penalty='l2', C=0.01, solver='liblinear').fit(Xdf, ydf)

    # Predict missing target_column values
    predicted_values = LR.predict(Xdf[missing_values])

    # Impute missing target_column values
    def impute_missing(df, predicted_values):
        count_unknown = df[target_column].value_counts()['Unknown']
        df.loc[df[target_column] == 'Unknown', target_column] = predicted_values[:count_unknown]
        return df

    df = impute_missing(df.copy(), predicted_values.copy())

    return df

#merge categories

def merge_infrequent_categories(df, column_to_check, infrequent_threshold, merge_with):
    """
    Merges infrequent categories in a specified column of a DataFrame with a given category.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to check.
        column_to_check (str): The name of the column to check for infrequent categories.
        infrequent_threshold (float): The frequency threshold below which categories are considered infrequent.
        merge_with (str): The category with which to merge the infrequent categories.

    Returns:
        pd.DataFrame: The DataFrame with infrequent categories merged.
    """
    # Calculate the frequency of each category in the specified column
    category_freq = df[column_to_check].value_counts(normalize=True)
    
    # Identify categories with frequency less than the threshold
    infrequent_categories = category_freq[category_freq < infrequent_threshold].index.tolist()
    
    # Merge infrequent categories with the specified category to merge with
    df[column_to_check] = df[column_to_check].replace(infrequent_categories, merge_with)
    
    return df

#Calculate 
def calculate_cv(series):
    #Calculate the coefficient of variation (CV) for a given series
    mean = series.mean()
    std_dev = series.std()
    cv = (std_dev / mean) * 100
    return cv

def impute_by_cv(df1, df2, group_by_column, cv_threshold=30, exclude_columns=[]):
    """
    Impute missing values in numeric columns based on coefficient of variation (CV).
    If CV < cv_threshold, impute by mean, otherwise impute by median.
    Exclude specified columns from imputation.
    """
    # Select numeric columns excluding specified columns
    numeric_columns = [col for col in df1.select_dtypes(include=[np.number]).columns.tolist() if col not in exclude_columns]

    # Iterate through numeric columns
    for column in numeric_columns:
        # Check for missing values
        if df1[column].isnull().sum() > 0:
            # Calculate CV
            cv = calculate_cv(df2[column].dropna())
            if cv < cv_threshold:
                df1 = mean_imputation_by_column(df1, df2, [column], group_by_column)
            else:
                df1 = median_imputation_by_column(df1, df2, [column], group_by_column)
    
    # Check for remaining missing values and impute accordingly, excluding specified columns
    remaining_missing_columns = [col for col in df1.columns[df1.isnull().any()] if col not in exclude_columns]
    numeric_remaining = df1[remaining_missing_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_remaining = df1[remaining_missing_columns].select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_remaining:
        df1 = mean_imputation(df1, df2, numeric_remaining)
    
    if categorical_remaining:
        df1 = mode_imputation(df1, df2, categorical_remaining)
    
    return df1