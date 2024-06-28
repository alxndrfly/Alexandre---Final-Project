import pandas as pd
import numpy as np
import ast
import warnings

# Suppress all warnings so the terminal doesnt go crazy
warnings.filterwarnings("ignore")

# Load the hotels dataset
hotels = pd.read_csv("datasets/raw/offerings.csv")

# Drop unnecessary columns
columns_to_drop = ['region_id', 'url', 'phone', 'details', 'type']
hotels.drop(columns=columns_to_drop, inplace=True)

# Function to extract values from the address column
def extract_address_info(address_str):
    address_dict = ast.literal_eval(address_str)
    return address_dict.get('locality', ''), address_dict.get('postal-code', ''), address_dict.get('street-address', '')

# Apply the function and create new columns
address_info = hotels['address'].apply(lambda x: pd.Series(extract_address_info(x)))
hotels[['locality', 'postal_code', 'street_address']] = address_info

# Drop the address column
hotels.drop(columns=['address'], inplace=True)

# Rename column for easier merging with reviews dataframe
hotels.rename(columns={'id': 'hotel_id'}, inplace=True)

# Calculate the mean of the hotel_class column
mean_hotel_class = hotels['hotel_class'].mean()

# Round the mean to the nearest 0.5
rounded_mean_hotel_class = np.round(mean_hotel_class * 2) / 2

# Replace missing values with the rounded mean
hotels['hotel_class'].fillna(rounded_mean_hotel_class, inplace=True)

# Reset the index
hotels.reset_index(drop=True, inplace=True)

# Export df to csv file for sql
hotels.to_csv('datasets/sql_cleaned/hotels.csv', index=False)

#############################################################

# Load the reviews dataset
reviews = pd.read_csv("datasets/raw/reviews.csv")

# Drop unnecessary columns
reviews.drop(columns=['id', 'num_helpful_votes'], inplace=True)

# Extract username from author column
reviews['author'] = reviews['author'].apply(lambda x: ast.literal_eval(x)['username'])

# Extract ratings and create new columns
ratings_df = reviews['ratings'].apply(lambda x: pd.Series(ast.literal_eval(x)))
reviews = pd.concat([reviews, ratings_df], axis=1)

# Drop the original ratings column and unnecessary columns
reviews.drop(columns=['ratings', 'check_in_front_desk', 'business_service_(e_g_internet_access)'], inplace=True)

# Rename columns
reviews.rename(columns={'date': 'date_review', 'offering_id': 'hotel_id'}, inplace=True)

# Drop rows with null values in the date_stayed column
reviews = reviews.dropna(subset=['date_stayed'])

# Convert date_stayed to MySQL format
reviews['date_stayed'] = reviews['date_stayed'].apply(lambda x: pd.to_datetime(x, format='%B %Y', errors='coerce').strftime('%Y-%m-%d') if pd.notna(pd.to_datetime(x, format='%B %Y', errors='coerce')) else None)
reviews = reviews.dropna(subset=['date_stayed'])

# Columns to fill
columns_to_fill = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']

# Fill missing values with the rounded mean of each column
for column in columns_to_fill:
    mean_value = reviews[column].mean()
    rounded_mean = round(mean_value)
    reviews[column].fillna(rounded_mean, inplace=True)

# Create full_text column and drop title and text columns
reviews['full_text'] = reviews['title'] + ' ' + reviews['text']
reviews.drop(columns=['title', 'text'], inplace=True)

# Move the full_text column to the first position
cols = ['full_text'] + [col for col in reviews if col != 'full_text']
reviews = reviews[cols]

# Calculate the mean rating for each row only for the specified columns and round to the nearest 0.1
specified_columns = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
reviews['mean_rating'] = reviews[specified_columns].mean(axis=1).round(1)

# Remove all types of quotes
reviews['full_text'] = reviews['full_text'].replace({r'[“”""]': ''}, regex=True)

# Reset the index
reviews.reset_index(drop=True, inplace=True)

# Export df to csv file for sql
reviews.to_csv('datasets/sql_cleaned/reviews.csv', index=False)