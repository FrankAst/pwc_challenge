import logging
import sys
import os

# Add the parent directories to the Python path when running directly
if __name__ == "__main__":
    # Add the src directory to the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    sys.path.insert(0, src_dir)
    sys.path.insert(0, project_root)

# Now we can use absolute imports
try:
    # Try relative imports first (when imported as part of package)
    from .load_and_merge import *
    from .FE_text import *
    from ..utils import remove_nulls
except ImportError:
    # Fall back to absolute imports (when running directly)
    from load_and_merge import *
    from FE_text import *
    from utils import remove_nulls, adjust_salary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths from root since its a package.
paths = ['data/people.csv',
         'data/salary.csv',
         'data/descriptions.csv']


# load_and_merge
# clean
# FE_text
# Fill other NaN
# FE_ratios

# VER FUNCTION COMPOSITION.

def main():
    """
    Main function to load and merge CSV files and print the result.
    """
    try:
        # Load data
        logger.info("Starting data loading and merging process... \n")
        logger.info(f"Files to merge: {paths}")
        
        df = load_and_merge(paths, "id")
        
        logger.info("Merge completed successfully!\n")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Delete rows with Salary NaN
        df = clean(df)
        logger.info("Data cleaning completed successfully!\n")
        logger.info(f"Cleaned dataset shape: {df.shape}")
        
        # Feature engineering on text data
        logger.info("Starting text feature engineering...\n\
                        Adding Part of speech features..")
        df[['noun_count', 'verb_count', 'adj_count', 'adv_count']] = df['Description'].apply(get_pos_tags)
        
        logger.info("Part of speech features added successfully!\n\
                        Starting filling missing data with description info...")        
        df = fill_missing_data(df)
        
        logger.info("Starting Job Title feature engineering...\n")
        
        df = apply_job_title_extraction(df)
        
        #logger.info("Job Title feature engineering completed successfully!\n")
        
        logger.info("Text feature engineering completed successfully!\n")
        logger.info(f"Dataset shape after text FE: {df.shape} \n")
        
        # delete rows with nulls mvp 1.0.1
        logger.info("Removing rows with null values...\n")
        df = remove_nulls(df)
        logger.info(f"Shape after removing nulls: {df.shape} \n")
        
        # Adjust salary values
        logger.info("Adjusting salary values...\n")
        df = adjust_salary(df)
        
        # Drop unnecessary columns
        logger.info("Dropping unnecessary columns: id, Job title and Description...\n")
        df.drop(columns=['id','Job Title', 'Description'], inplace=True)
        
        # Save the final DataFrame to a CSV file
        output_path = 'data/cleaned_data/final_dataset.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Final dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    
    main()