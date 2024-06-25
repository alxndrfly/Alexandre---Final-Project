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

2 - Classify user reviews in different categories to plot by hotel which aspects to improve first. ( specific aspects such as cleanliness, staff, amenities, location, value for money, food quality, etc. ) ( Aspect Sentiment Score: Calculate sentiment scores for each aspect to identify areas of strength and weakness. )

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

All the detailed steps can be found as comments inside the python scripts contained in this git repository.


# Cloud deployment with AWS

Created an EC2 instance running ubuntu 24.04
Created a mysql database in RDS and linked to my EC2 instance
