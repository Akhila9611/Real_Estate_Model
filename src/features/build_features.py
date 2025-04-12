import pandas as pd

def create_dummy_vars(df):
    """
    Create dummy variables for categorical features.
    Args:
        df (pd.DataFrame): The dataset with categorical columns.
    Returns:
        pd.DataFrame: The dataset with dummy variables.
    """
    # Create dummy variables for 'property_type_Bunglow' and 'property_type_Condo'
    df = pd.get_dummies(df, columns=['property_type_Bunglow', 'property_type_Condo'], dtype=int)
    
    # Store the processed dataset in 'data/processed'
    df.to_csv('data/processed/Processed_final.csv', index=None)
    
    return df

def separate_features_target(df):
    """
    Separate the features and target variable.
    Args:
        df (pd.DataFrame): The processed dataset.
    Returns:
        X (pd.DataFrame): Feature set
        y (pd.Series): Target variable
    """
    X = df.drop('price', axis=1)  # Dropping the target column 'price' from features
    y = df['price']  # The 'price' column is the target variable
    return X, y
