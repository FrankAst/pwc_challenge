import spacy
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------## TEXT FUNCTIONS ##--------------------------------------------------------------#

# Function to generate part of speech columns using Spacy.
def get_pos_tags(text):
    if pd.isna(text):
        return pd.Series([0, 0, 0, 0])
    
    doc = nlp(text)
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    return pd.Series([pos_counts["NOUN"], pos_counts["VERB"], pos_counts["ADJ"], pos_counts["ADV"]])



# -------------------------------------------## LLM OpenAI FUNCTIONS ##---------------------------------------------------------#


# Function to extract missing values from description using OpenAI's GPT model.
# Model chose is gpt-4o-mini, which is optimized for cost and speed, with $0.15 per MIT and $0.60 per MOT.

def extract_data_llm(row):
    
    description = row['Description']
    
    if pd.isna(description):
        return row
    
    # Check which fields are missing
    missing_fields = []
    if pd.isna(row['Age']):
        missing_fields.append('Age')
    if pd.isna(row['Gender']):
        missing_fields.append('Gender')
    if pd.isna(row['Education Level']):
        missing_fields.append('Education Level')
    if pd.isna(row['Job Title']):
        missing_fields.append('Job Title')
    
    if not missing_fields:
        return row
    
    prompt = f"""Extract the following information from this job description. Return ONLY a JSON object.

    MISSING FIELDS TO EXTRACT: {', '.join(missing_fields)}

    CRITICAL INSTRUCTIONS:
    - Return raw JSON only, no ```json``` blocks
    - If information is NOT FOUND, return null for that field
    - Do NOT guess or make up information
    - Only extract information that is explicitly stated

    Example input:
    1. "I am a 29-year-old female Marketing Coordinator with a Bachelor's degree in Communications and 
    four years of experience in digital marketing. My expertise includes social media strategy, content creation,
    email marketing campaigns, and brand management. I specialize in developing engaging content across multiple 
    platforms, analyzing campaign performance metrics, and coordinating with creative teams to deliver compelling 
    marketing materials. My role involves managing social media accounts, creating marketing collateral, and supporting 
    lead generation initiatives. I am passionate about staying current with emerging digital trends and continuously 
    expanding my knowledge of marketing automation tools to drive brand awareness and customer engagement."
    
    Job examples:
    "Senior Software Engineer", "Data Scientist", "Marketing Manager", "Supply chain Analyst"
    "Senior Product Manager", "UX Designer", "Business Analyst", "Sales Executive", "Director of Sales",
    "Chief Technology Officer", "Human Resources Manager", "Financial Analyst", "Project Manager", "Content Strategist"
    
    EXAMPLE OUTPUTS:
    // All fields found:
    {{"Age": 33, "Gender": "Male", "Education Level": "Bachelor's", "Job Title": "Senior Software Engineer"}}

    // Age not found:
    {{"Age": null, "Gender": "Female", "Education Level": "Master's", "Job Title": "Data Scientist"}}

    // Only job title found:
    {{"Age": null, "Gender": null, "Education Level": null, "Job Title": "Marketing Manager"}}


    Description: {description}

    JSON:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        
        # Get the response content
        response_text = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Parse JSON response
        result = json.loads(response_text)
        
        
        # Update row with extracted values
        updated_row = row.copy()
        
        for field in missing_fields:
            if field in result:
                value = result[field]
                # Only update if we got a valid value (not null, not empty string)
                if value is not None and value != "" and value != "Unknown":
                    updated_row[field] = value
        
        return updated_row
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON parsing error for row {row.name}: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return row
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return row
    
# Function fill the missing data with the LLM output.
def fill_missing_data(df):
    
    """
    Fill missing data in the DataFrame using OpenAI's GPT model.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with missing values.
    
    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    # Columns to fill
    target_cols = ['Age', 'Gender', 'Education Level', 'Job Title']
    missing_mask = df[target_cols].isna().any(axis=1)
    
    print(f"Processing {missing_mask.sum()} rows with missing data...")
    
   # Get the data from description column:
    df_updated = df.copy()
    
    for idx in df[missing_mask].index:
                
        df_updated.loc[idx] = extract_data_llm(df.loc[idx])
    
    return df_updated


    
