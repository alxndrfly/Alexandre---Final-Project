import time

# Record the start time
start_time = time.time()

import pandas as pd
import warnings

# Suppress all warnings so the terminal doesnt go crazy
warnings.filterwarnings("ignore")

reviews = pd.read_csv('datasets/model_data/tagged_reviews.csv')
ids = pd.read_csv('datasets/sql_cleaned/reviews.csv')

reviews['hotel_id'] = ids['hotel_id']
reviews['sentiment'] = reviews['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})

reviews.to_csv('datasets/model_data/tagged_reviews.csv', index=False)


def keep_top_hotels_by_reviews(df, num_hotels):
    # Count the number of reviews per hotel
    hotel_counts = df['hotel_id'].value_counts()

    # Get the top `num_hotels` hotel IDs with the most reviews
    top_hotels = hotel_counts.nlargest(num_hotels).index

    # Filter the DataFrame to keep only the rows of these top hotels
    truncated_df = df[df['hotel_id'].isin(top_hotels)]
    
    return truncated_df

# Number of hotels to keep
num_hotels_to_keep = 100

# Get the truncated DataFrame
short_reviews = keep_top_hotels_by_reviews(reviews, num_hotels_to_keep)

# Reset index
short_reviews.reset_index(drop=True, inplace=True)

# Store both mapped datasets for model training
short_reviews.to_csv('datasets/model_data/truncated_train_data.csv', index=False)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")