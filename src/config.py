# Configuration constants for my models

# Features:
CATEGORICAL_FEATURES = ['Gender', 'Education Level', 'Seniority', 'Area', 'Role']
NUMERICAL_FEATURES = ['Age', 'Years of Experience', 'noun_count', 'verb_count', 'adj_count', 'adv_count']
TARGET_COLUMN = 'Salary'

# Model parameters:
SEED = 37
TEST_SIZE = 0.2

# Dataset Path
FILE_PATH = 'data/cleaned_data/final_dataset.csv'

