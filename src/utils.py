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

def adjust_salary(df):
    """
    Adjusts rows where salary is too small.
    """
    # Adjust rows where salary is too small
    df.loc[df['Salary'] < 1000, 'Salary'] = df['Salary'] * 100
    
    return df



#################################### Report aux plotting functions ####################################

def plot_salary_distribution(df):
    """
    Plots the distribution of the 'Salary' column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'Salary' column.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['Salary'], bins=40, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Salary')
    plt.xlabel('Salary ($)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
def plot_correlation_matrix(df):
    """
    Plots the correlation matrix for numerical variables in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame for which to plot the correlation matrix.
    """
    import matplotlib.pyplot as plt
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_cols].corr()

    # Create correlation plot
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(numerical_cols)), numerical_cols, rotation=45)
    plt.yticks(range(len(numerical_cols)), numerical_cols)
    plt.title('Correlation Matrix of Numerical Variables')

    # Add correlation values as text
    for i in range(len(numerical_cols)):
        for j in range(len(numerical_cols)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                     ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()
    
def plot_salary_boxplot_by_category(df, category_column):
    """
    Plots a colorful box plot for salary distribution by a specified categorical feature, excluding null categories.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing 'Salary' and the specified categorical column.
    category_column (str): The name of the categorical column to group by.
    """
    import matplotlib.pyplot as plt
    
    # Drop rows with null values in the specified category column or 'Salary'
    df = df.dropna(subset=[category_column, 'Salary'])
    
    # Create a colorful box plot for salary distribution by the specified category
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = df[category_column].unique()
    box_data = [df[df[category_column] == category]['Salary'] for category in categories]
    
    # Define custom colors for the boxes
    colors = plt.cm.tab10.colors[:len(categories)]
    
    bp = ax.boxplot(box_data, labels=categories, patch_artist=True)
    
    # Apply different colors to each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(f'Salary Distribution by {category_column}', fontsize=16)
    ax.set_xlabel(category_column, fontsize=12)
    ax.set_ylabel('Salary ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()