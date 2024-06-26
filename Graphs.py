import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns

#plot variables with less than 60% missing values
def plot_missing_values(data, threshold):
    nan_below_threshold = [k for k, v in dict(data.isna().mean()).items() if 0<v < threshold]
    
    if len(nan_below_threshold) == 0:
        print("No variables found with missing values below the threshold.")
        return
    
    nan_below_threshold_df = pd.DataFrame({
        'Variable': nan_below_threshold,
        'Missing Percentage': data[nan_below_threshold].isna().mean().sort_values(ascending=False) * 100
    })

    plt.figure(figsize=(10, 6))
    plt.barh(nan_below_threshold_df['Variable'], nan_below_threshold_df['Missing Percentage'], color='purple')
    plt.xlabel('Missing Percentage (%)')
    plt.ylabel('Variable')
    plt.title(f'Variables with missing values below {threshold*100}%')
    plt.gca().invert_yaxis()  
    plt.show()

#plot histogram
def plot_histogram(data, variable):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[variable], bins=30)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {variable}')
    plt.show()

#plot bar charts for categorical variables
def plot_categorical_distribution(data, categorical_variables):
    sns.set(style="whitegrid")
    num_variables = len(categorical_variables)
    num_rows = (num_variables - 1) // 3 + 1
    plt.figure(figsize=(15, num_rows * 5))
    
    for i, variable in enumerate(categorical_variables, 1):
        plt.subplot(num_rows, 3, i)
        sns.countplot(data=data, x=variable, palette="pastel")
        plt.title(f"Distribuția variabilei {variable}")
        plt.xlabel(variable)
        plt.ylabel("Frecvență")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

#plot box-plot
def plot_boxplot(data, x_variable, y_variable):
    # Plot the boxplot
    sns.boxplot(x=x_variable, y=y_variable, data=data, palette='pastel')
    plt.title(f'Box Plot: {y_variable} for each category of {x_variable}')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.show()

#plot correlation matrix_heatmap
def plot_correlation_heatmap(data):
    # Calculate the correlation matrix
    corr_df = data.corr()
    
    # Select only correlations with coefficient higher than 0.01
    corr_df = corr_df[(abs(corr_df) > 0.01).any(axis=1)]
    corr_df = corr_df.loc[:, (abs(corr_df) > 0.01).any(axis=0)]
    
    # Plot the heatmap
    plt.figure(figsize=(20, 5))
    sns.heatmap(corr_df, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_pairplot(data):
    sns.set(rc={'figure.figsize':(8, 8)})
    sns.pairplot(data, plot_kws=dict(marker="+", linewidth=1), palette="Paired")
    plt.show()


def plot_frequency(df):
    # Calculate the frequency of each race category in the patient_race variable
    race_freq = df['patient_race'].value_counts()

    # Display a bar chart for race frequency
    race_freq.plot(kind='bar', color='purple')

    # Add title and axis labels
    plt.title('Bar Chart for Race Frequency')
    plt.xlabel('Races')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

def plot_numerical_features(train_data, feature_importance, target_column='treatment_pd', threshold=0.3):
    """
    Plot bar charts for numerical features with importance greater than the threshold.

    Args:
        train_data (pd.DataFrame): Training dataset with features and the target column.
        feature_importance (pd.DataFrame): DataFrame containing feature importances with index as feature names.
        target_column (str): The name of the target column in the training dataset.
        threshold (float): Importance threshold to select important features.

    Returns:
        Graphics
    """
    # Select numerical features with importance greater than the threshold
    numerical_features = feature_importance[(feature_importance['importance'] > threshold) & (train_data.dtypes != 'object')].index.tolist()

    # Define a custom color palette from red to green
    n = len(train_data[target_column])
    

    # Plot bar chart for each important numerical feature
    for feature in numerical_features:
        plt.figure(figsize=(12, 6))

        # Calculate the number of bins based on unique values
        num_unique_values = len(train_data[feature].unique())
        max_bins = min(num_unique_values, 5)

        # Define bins automatically
        bins = pd.qcut(train_data[feature], q=max_bins, duplicates='drop')

        # Calculate mean target value for each bin
        means = train_data.groupby(bins)[target_column].mean().reset_index()

        # Sort bins by mean target value in descending order
        means.sort_values(by=target_column, ascending=False, inplace=True)

        # Remove bins with zero frequency
        means = means[means[target_column] != 0]

        # Convert bins to string representation of interval
        interval_labels = [f'{interval.left}-{interval.right}' for interval in means[feature]]

        # Plot bar chart
        sns.barplot(x=interval_labels, y=target_column, data=means, palette=sns.color_palette("RdYlGn", n_colors=len(means)))

        # Set title and labels
        plt.title(f'Mean {target_column} by {feature} intervals')
        plt.xlabel(feature)
        plt.ylabel(target_column)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        plt.show()

def plot_categorical_features(train_data, target_column, feature_importance, threshold=0.3):
    """
    Plot bar charts for categorical features with importance greater than the threshold.

    Args:
        train_data (pd.DataFrame): Training dataset with features and the target column.
        target_column (str): The name of the target column in the training dataset.
        feature_importance (pd.DataFrame): DataFrame containing feature importances with index as feature names.
        threshold (float): Importance threshold to select important features.

    Returns:
        Graphics
    """
    # Select categorical features with importance greater than the threshold
    important_categorical_features = feature_importance[(feature_importance['importance'] > threshold) & (train_data.dtypes == 'object')].index.tolist()
    print("Categorical features with importance greater than {}:\n".format(threshold), important_categorical_features)

    # Plot bar chart for each important categorical feature
    for feature in important_categorical_features:
        plt.figure(figsize=(20, 8))
        means = train_data.groupby(feature)[target_column].mean().reset_index()
        means.sort_values(by=target_column, ascending=False, inplace=True)  # Sort by target_column
        sns.barplot(x=feature, y=target_column, data=means, palette=sns.color_palette("RdYlGn", n_colors=len(means)))
        plt.title(f'Mean {target_column} by {feature}')
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)  # Adjust the fontsize of the x-axis labels
        plt.show()