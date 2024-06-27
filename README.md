# Alexandre-Final-Project
Here is a Trello screenshot of the working steps i will be taking to complete this project.
![image](https://github.com/alxndrfly/Alexandre-Final-Project/assets/135460292/2db84cb3-481c-4ffe-b18a-7d3573c663a5)


# Business problem

This project assumes we are a business analyst for a hotel review management company.
This company uses AI and LMs to generate answers based on the user's review and repplies on all platforms for the hotel directly.

The company currently operates in France, Belgium and Spain and wants to expand its customer base and growth.

Our goal is to perform sentiment analysis on the hotel reviews and classify them. Then we build a valuable report for the hotel owner containing various insights ( increase in positive reviews, increase in reviews, time saved by using our service, money saved compared to the plan cost, how quickly the reviews were repplied to... )

# Dataset introduction

Using the trip advisor hotel reviews dataset from kaggle ( https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews/data?select=reviews.csv ).
This dataset contains two csv files : 1. ( offerings.csv ) Hotel information with their ids, number of stars and addresses. 2. ( reviews.csv ) tripadvisor hotel booker's reviews of hotels spanning all accross the us.

All the hotels mentionned in this analysis are located in different states of the United States of America.
List of localities of the hotels : New York City, Houston, San Antonio, Los Angeles, San Diego, San Francisco, Dallas, Austin, Indianapolis, Phoenix, Charlotte, Chicago, Columbus, Denver, Jacksonville, Memphis, Washington DC, Seattle, Fort Worth, Philadelphia, El Paso, Boston, San Jose, Baltimore, Detroit.

# Approach

1 - Perform sentiment analysis to classify user reviews as positive or negative ( this gives us the ratio of good/bad review for each hotel and will be used as business insight )

2 - Classify user reviews in different categories using Aspect-Based Sentiment Analysis (ABSA) to plot by hotel which aspects to improve first. ( specific aspects such as cleanliness, staff, amenities, location, value for money, food quality, etc. ) ( Aspect Sentiment Score: Calculate sentiment scores for each aspect to identify areas of strength and weakness. )

3 - Provide a broad market analysis and insights of positive vs negative reviews and what categories are the most liked and disliked by customers etc... as a tableau dashboard.

4 - Generate an interactive dashboard for a specific hotel, aiming to provide insights ( returning customer count, number of reviews compared to past month, % of positive reviews, average speed of answer... ) to customers ( hotels ) and increase retention ( for our company ) as a web app.

5 - BONUS - Give suggestions to the business owner on which hotel to pitch next about our services.

# Exploratory Data Analysis

We are working with 878'561 reviews of 4,333 hotels.

These 878'561 reviews were left by 536'952 unique customers. Therefore 61% of trip advisor hotel reviewers leave more than 1 review and likely book more than 1 hotel on the platform. We will be tracking returning reviewers for each hotel.


![image](https://github.com/alxndrfly/Alexandre-Final-Project/assets/135460292/5a425ae9-ba77-4baf-808a-99d5d365e952)

Most hotel reviews left by tripadvisor users are from other devices ( desktop, laptop, tablet... ) than mobile phones


![image](https://github.com/alxndrfly/Alexandre-Final-Project/assets/135460292/f6758c42-7850-4046-8d55-e9ba54cb6e84)

Most users leave only 1 review.

![image](https://github.com/alxndrfly/Alexandre-Final-Project/assets/135460292/34bd72d7-f216-4bf0-938a-a3b6c295e273)

Here is a log scale to be able to visualise that there are in fact users that leave large amounts of reviews. It is in fact very rare.

![image](https://github.com/alxndrfly/Alexandre-Final-Project/assets/135460292/85ed25dd-057f-412f-8b05-66a7d7a5aac3)

Overall High Ratings: The majority of the ratings are 4.0 and 5.0 across all metrics. This indicates that the hotels generally receive high ratings from reviewers.
Important to keep in mind that our text review data will be imbalanced when performing sentiment analysis.

# Extract Transform Load

Before running ETL.py ensure you have ("datasets/raw/offerings.csv") and ("datasets/raw/reviews.csv") in the right directory.

All the detailed steps can be found as comments inside the python script ETL.py contained in this git repository.
This script takes the raw data, transforms it and stores it in a directory ('datasets/sql_cleaned/file_name.csv'), for later sql injection.

For hotels dataset :
- Load the hotels dataset.
- Drop columns with unnecessary info : ['region_id', 'url', 'phone', 'details', 'type'].
- Extract and store as new columns the locality, postal code and street address contained in the column address.
- Drop the address column.
- Rename id to hotel_id to have a primary key between hotels.csv and reviews.csv.
- Fill the class column's (n of hotel stars) missing values with the mean (rounded to the nearest 0.5) of the hotels class.
- Reset the index and save the hotel.csv in the sql_cleaned dir.

For reviews dataset :
- Load the reviews dataset.
- Drop columns with unnecessary info : ['id', 'num_helpful_votes'].
- Extract all the ratings in the column ratings and store each in its own column.
- Drop columns with unnecessary info : ['ratings', 'check_in_front_desk', 'business_service_(e_g_internet_access)'].
- Rename date and offering_id to date_reviews and hotel_id (primary key).
- Drop rows with null values in the date_stayed column (needed for our dashboard and analysis).
- Convert date_stayed to MySQL format.
- Fill missing values in the ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms'] rating columns with the rounded mean of each column.
- Merged the column title and text into full_text.
- Calculate the mean rating for each rating row only, round to the nearest 0.1 and make a new column mean_rating.
- Reset the index and save the reviews.csv in the sql_cleaned dir.

Used to build a tableau dashboard showcasing the evaluation and overview of the entire dataset.

  # Pseudo-labeling the data

I decided to take a shortcut by using roBERTa ('siebert/sentiment-roberta-large-english') to label my data. 
Since i have 800'000+ rows, labeling it by hand would required too much time or would be costly.
Doing so allows me to finish the project in the required time window and include all other important aspects to showcase my skills.
The script label_data_roBERTa.py takes in the ETL'd data from the directory sql_cleaned, preprocess' it for roBERTa, performs sentiment analysis and stores a tagged_reviews.csv in the model_data directory.
We will use this new csv, containing only the non processed text, hotel_id and the POSITIVE or NEGATIVE labels to train our own models.

  # Last ETL step

truncate_train_data.py does two things:

1. Add the hotel_id to the tagged_reviews.csv file, change the POSITIVE and NEGATIVE to 1 and 0 respectively and saves the file.
2. Take the tagged_reviews.csv file, selects the 100 hotels with the most reviews and saves a new file truncated_train_data.csv.
   That way we have a file containing 178'431 reviews from the top most reviewed hotels instead of 800'000+ to train faster.


# Model training and testing for sentiment analysis

I trained and tested 3 models for sentiment analysis (positive = 1, negative = 0)
1. Logistic Regression
2. Support Vector Machine
3. Random Forest Classifier

In all 3 cases i used :
- MLFlow locally to log my metrics and best parameters for each class and save my model, roc curve, confusion matrix and classification report as artifacts.
- Train test split of 80-20.
- Pipeline and cross validation to test different hyperparameters.
- TFIDFVectoriser to convert text to numerical data.
- SMOTE to balance the negative class. Since our dataset is imbalanced (82% positive and 18% negative).

First i used only 10'000 data points from the truncated dataset to get the best parameters with faster training times, then i trained all 3 models with the entire truncated dataset to compare their metrics.

In the git repo you will find : random_forest.py, svmachine.py and logistic_regression.py.
I recomend manually changing the amount of rows you want to use to test the models and type in the hyperparameters you wish to test as i decided not to include every single time i change either of these as a separate .py file. The current scripts train the models with all the truncated dataset and with the best hyperparameters from testing with 10'000 rows only. It gest logged in MLFlow for you to see and compare. Have fun with it.

# Selecting the best model based on performance metrics

From the model's metrics we can see that where some models fall short is in the recall for the negative reviews. It is important that we can correctly identify both classes to build reports for our customers. Therefore we will pick the model that has the highest recall and f1 score for both negative and positive classes.

showing best metrics for each model trained with truncated data

picking model

explaining stenghts

explain limitations

# Aspect-Based Sentiment Analysis (ABSA)

Here we will pick a hotel of interest and perform ABSA in order to generate a personalised report with useful insights. The goal is to provide clear value to our customers and increase retention.

# Cloud deployment with AWS

TO FILL

# 
