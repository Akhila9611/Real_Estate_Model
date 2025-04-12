import pandas as pd

def load_data():
    """
    Load the raw dataset from the 'data/raw' folder.
    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    dataset_path = 'data/raw/final.csv'  # Path to the raw data
    df = pd.read_csv(dataset_path)
    return df

def inspect_data(df):
    """
    Display the first and last few rows, as well as the shape of the data.
    Args:
        df (pd.DataFrame): The dataset to inspect.
    """
    print("First few rows:")
    print(df.head())
    
    print("\nLast few rows:")
    print(df.tail())
    
    print("\nShape of the dataset:")
    print(df.shape)
