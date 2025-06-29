import spacy
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import logging

# Suppress HTTP status code from console output.
logging.getLogger("httpx").setLevel(logging.WARNING)

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
# Chosen model is gpt-4o-mini, which is optimized for cost and speed, with $0.15 per MIT and $0.60 per MOT.

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
    
# Function to fill the missing data with the LLM output.
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


# Function to explode job title into: seniority, area, role. Missing values are fetched from description using OpenAI's GPT model.
# Chosen model is gpt-4o-mini, which is optimized for cost and speed, with $0.15 per MIT and $0.60 per MOT.

def extract_job_title_info(row):
    """Extract seniority, area, and role from job title"""
    
    
    job_title = row['Job Title']
    
    if pd.isna(job_title):
        return row
    
    years_exp = row.get('Years of Experience', None)
    
    prompt = f"""Extract the following information from this job title. Return ONLY a JSON object.

    JOB TITLE: {job_title}
    YEARS OF EXPERIENCE: {years_exp}

    CRITICAL VALIDATION RULES:
    - The job title MUST be a realistic, professional job position that exists in the real world
    - If the job title is fake, exaggerated, nonsensical, or contains inappropriate content, return null for ALL fields
    - Do NOT extract from obviously fake titles like "CEO of World", "King of Universe", "Supreme Leader", etc.
    - Do NOT extract from titles with excessive punctuation, all caps, or unprofessional language
    - Do NOT extract from gibberish, random characters, or made-up words
    - Do NOT extract from titles that are clear references to FANTASY characters. ()
    
    EXAMPLES OF INVALID JOB TITLES (return null for all fields):
    - "jajasjkdkasdjkaj" 
    - "CEO OF WORLD!"
    - "King of Universe"
    - "Supreme Leader"
    - "Data Wizard"
    - "Superhero"
    - "Master of Everything"
    - "God of Code"
    - "Emperor of Sales"
    - "Ninja Developer"
    - "Rock Star Engineer"
    - "Unicorn Designer"
    - "asdfghjkl"
    - "123456"
    - "!!!MANAGER!!!"
    - "CEO OF EVERYTHING"
    
    EXAMPLES OF VALID JOB TITLES:
    - "CEO" (real executive role)
    - "Data Engineer"
    - "Software Developer" 
    - "Marketing Manager"
    - "Product Manager"
    - "Senior Consultant"
    - "Director of Sales"
    - "Chief Technology Officer"

    EXTRACTION RULES (only if job title is valid):
    - Return raw JSON only, no ```json``` blocks
    - Only extract information that is explicitly stated
    
    SENIORITY RULES:
    - If the job title contains a seniority level (e.g., "Senior", "Junior", "Lead", "Principal", etc), extract it
    - If no seniority level is present, return "rule_based" (I will handle this programmatically)
        
    AREA RULES:
    - If the job title contains a specific area (e.g., "Software Engineering", "Data Science", "Marketing", etc), extract it
    - Try to group similar areas together (e.g., "Software Engineering" and "Software Development" as "Software/IT")
    - Avoid being too specific, use broader categories when possible (e.g., "Consulting" instead of "Management Consulting")
    - Data analysts and data scientists should be grouped under "Data Science"
    
    Example outputs:
    - "CEO" -> {{"seniority": "C-level", "area": "Executive", "role": "CEO"}}
    - "Recruiter" -> {{"seniority": "rule_based", "area": "Human Resources", "role": "Recruiter"}}
    - "Senior Consultant" -> {{"seniority": "Senior", "area": "Consulting", "role": "Consultant"}}
    - "Data Scientist" -> {{"seniority": "rule_based", "area": "Data Science", "role": "Scientist"}}
    - "Software Engineer" -> {{"seniority": "rule_based", "area": "Software/IT", "role": "Engineer"}}
    - "jajasjkdkasdjkaj" -> {{"seniority": null, "area": null, "role": null}}

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
        
        # Additional validation - check for obvious fake patterns
        job_title_upper = job_title.upper()
        fake_patterns = [
            'OF WORLD', 'OF UNIVERSE', 'OF EVERYTHING', 'SUPREME', 'MASTER OF', 
            'GOD OF', 'KING', 'EMPEROR', 'WIZARD', 'NINJA', 'ROCK STAR',
            'UNICORN', '!!!', 'CEO OF', 'PRESIDENT OF THE WORLD', 'ORC', 'ELF', 'DRAGON'
        ]
        
        is_fake = any(pattern in job_title_upper for pattern in fake_patterns)
        
        # Check for excessive punctuation or all caps (except normal acronyms)
        has_excessive_punctuation = job_title.count('!') > 1 or job_title.count('?') > 0
        is_all_caps_fake = job_title.isupper() and len(job_title) > 5 and not job_title.replace(' ', '').isalpha()
        
        # Validate that we got meaningful results
        area = result.get('area', None)
        role = result.get('role', None)
        
        # If both area and role are null, or if we detected fake patterns, the job title is invalid
        if (area is None and role is None) or is_fake or has_excessive_punctuation or is_all_caps_fake:
            # Return the original row but add a validation error flag
            updated_row = row.copy()
            updated_row['Seniority'] = None
            updated_row['Area'] = None
            updated_row['Role'] = None
            updated_row['_validation_error'] = f"Invalid job title: '{job_title}'"
            return updated_row
        
        # Handle rule-based seniority determination
        if result.get('seniority') == 'rule_based':
            if pd.notna(years_exp):
                if years_exp >= 5:
                    result['seniority'] = 'Senior'
                elif years_exp >= 0 and years_exp <= 4:
                    result['seniority'] = 'Junior'
                else:
                    result['seniority'] = None
            else:
                result['seniority'] = None
        
        # Update row with extracted values
        updated_row = row.copy()
        updated_row['Seniority'] = result.get('seniority', None)
        updated_row['Area'] = area
        updated_row['Role'] = role
        
        return updated_row
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON parsing error for row {row.name}: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return row
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return row
 
 
# Standardize Areas & Roles obtained through the LLM, to reduce noise in the dataset.
def aggregate_categories(value, category_type):
    """Aggregate areas and roles into broader categories"""
    
    if pd.isna(value) or value == 'rule_based':
        return 'Other'
    
    if category_type == 'area':
        mapping = {
            'Software/data': ['Software/IT', 'Data Science', 'Engineering', 'Web Design', 'Technical Writing'],
          
            'Business': ['Business Analysis', 'Business Development', 'Business Operations', 
                        'Consulting', 'Executive'],
            
            'Sales & Marketing': ['Marketing', 'Sales', 'Sales/Marketing', 'Public Relations', 
                                'Content Production', 'Creative'],

            'Finance & Operations': ['Finance', 'Accounting', 'Operations', 'Supply Chain'],
            
            'Product & Design': ['Product Management', 'Product Development', 'Design', 
                            'User Experience'],

            'People & Support': ['Human Resources', 'Training', 'Customer Service', 
                            'Customer Success', 'Customer Support', 'Account Management'],

            'Management': ['Project Management', 'Administration', 'Events'],

            'Other': ['Research', 'Data Entry']
        }
    
    elif category_type == 'role':
        mapping = {
            
            'TopExecs': ['CEO', 'CFO', 'COO', 'CTO', 'CIO', 'President', 'Vice President', 'VP', 'Executive Vice President', 
                        'Senior Vice President', 'Chief Executive Officer',],
            
            'Leadership': ['Director', 'Executive', 'Technology Officer'],
            
            'Management': ['Manager', 'Operations Manager', 'Product Manager', 'Project Manager', 
                          'Account Manager', 'Product Marketing Manager', 'Product Management'],
            
            'Analysis': ['Analyst', 'Business Analyst', 'Data Scientist', 'Scientist', 'Researcher', 
                        'Quality Assurance Analyst'],
            
            'Engineer': ['Engineer', 'Developer', 'Engineering', 'Architect'],
            
            'Support': ['Coordinator', 'Specialist', 'Support Specialist', 'Assistant', 'Support', 
                       'Clerk', 'HR'],
            
            'Individual Contributor': ['Associate', 'Representative', 'Rep', 'Generalist', 'Recruiter', 
                                      'Accountant', 'Consultant', 'Advisor', 'Account Executive'],
            
            'Creative': ['Designer', 'Product Designer', 'Graphic Designer', 'Copywriter', 'Writer', 
                        'Producer']
        }
    
    else:
        raise ValueError("category_type must be 'area' or 'role'")
    
    # Find matching category
    for category, items in mapping.items():
        if value in items:
            return category
    
    return 'Other'   
    
    
# Function to apply the job title extraction function to the DataFrame.
def apply_job_title_extraction(df):
    
    """Apply job title extraction to entire dataframe"""

    # Initialize new columns if they don't exist
    for col in ['Seniority', 'Area', 'Role']:
        if col not in df.columns:
            df[col] = None

    # Apply extraction
    for idx in df.index:
        if idx % 50 == 0:
            print(f"Processing row {idx}...")
        df.loc[idx] = extract_job_title_info(df.loc[idx])
    
    # Aggregate areas and roles into broader categories
    df['Area'] = df['Area'].apply(lambda x: aggregate_categories(x, 'area'))
    df['Role'] = df['Role'].apply(lambda x: aggregate_categories(x, 'role'))
    

    return df



