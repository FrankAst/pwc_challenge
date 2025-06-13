import logging
from load_and_merge import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

paths = ['../data/people.csv',
         '../data/salary.csv',
         '../data/descriptions.csv']


def main():
    """
    Main function to load and merge CSV files and print the result.
    """
    try:
        logger.info("Starting data loading and merging process...")
        logger.info(f"Files to merge: {paths}")
        
        merged_df = load_and_merge(paths, "id")
        
        logger.info("Merge completed successfully!")
        logger.info(f"Final dataset shape: {merged_df.shape}")
        logger.info(f"Columns: {list(merged_df.columns)}")
        
        logger.info("First 5 rows of merged data:")
        logger.info(f"\n{merged_df.head(5)}")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    
    main()