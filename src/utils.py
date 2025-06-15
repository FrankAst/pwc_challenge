import pandas as pd

def remove_nulls(df):
    """
    Remove rows with any null values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame from which to remove null values.
    
    Returns:
    pd.DataFrame: DataFrame with rows containing null values removed.
    """
    # Check for duplicated rows and remove them
    df = df.drop_duplicates()
    
    return df.dropna()