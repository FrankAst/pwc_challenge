import pandas as pd
from typing import List


def load_and_merge(file_paths: List[str], join_column: str) -> pd.DataFrame:
    """
    Load CSV files and left join them in the order they are provided.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of CSV file paths to load and merge
    join_column : str
        Column name to join on
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    result = pd.read_csv(file_paths[0])
    for path in file_paths[1:]:
        df = pd.read_csv(path)
        result = pd.merge(result, df, on=join_column, how='left')
    return result


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataframe by dropping rows with NaN values in target column.
    
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.dropna(subset=['Salary'])

    
    return df.dropna(subset=['id'])