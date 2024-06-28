import streamlit as st
import joblib
import openai
from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer
import torch
import seaborn as sns
from collections import Counter

# Load the vectorizer and the model from app_artifacts directory
vectorizer = joblib.load('app_artifacts/vectorizer.joblib')
model = joblib.load('app_artifacts/model.joblib')

# Set your OpenAI API key
openai.api_key = ''

client = OpenAI(api_key=openai.api_key)

def get_chatgpt_response(review):
    messages = [
        {
            "role": "system",
            "content": "As a representative of a hotel, please provide a short, courteous but friendly response to the following review:"
        },
        {
            "role": "user",
            "content": review
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content

def get_summary_response(review):
    messages = [
        {
            "role": "system",
            "content": "Summarize the following hotel review into a short report with bullet points about the positive keywords and negative keywords:"
        },
        {
            "role": "user",
            "content": review
        }]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content

def load_dataset(dataset_name):
    return pd.read_csv(f'app_artifacts/{dataset_name}')

################################################################################

def plot_summary_statistics(df):
    st.write("###")
    hotel_name = df['name'].iloc[0]
    st.write(f"# {hotel_name}")
    st.write("###")
    st.write("## General Metrics")

    # Calculate general metrics
    sentiment_counts = df['sentiment'].value_counts(normalize=True)
    positive_ratio = sentiment_counts[1] * 100 if 1 in sentiment_counts else 0
    overall_mean_rating = df['mean_rating'].mean()
    overall_mean_cleanliness = df['cleanliness'].mean()
    overall_mean_sleep_quality = df['sleep_quality'].mean()
    overall_mean_rooms = df['rooms'].mean()
    overall_mean_service = df['service'].mean()
    overall_mean_value = df['value'].mean()
    
    # Calculate Returning Customer Rate
    total_customers = df['author'].nunique()
    returning_customers = df['author'].value_counts().loc[lambda x: x > 1].count()
    returning_customer_rate = (returning_customers / total_customers) * 100

    # Most loyal customer ever
    most_loyal_customer_ever = df['author'].value_counts().idxmax()
    most_loyal_customer_count_ever = df['author'].value_counts().max()

    # Display general metrics side by side
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Positive Reviews Ratio", value=f"{positive_ratio:.2f}%")
    with col2:
        st.metric(label="Mean Rating", value=f"{overall_mean_rating:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Returning Customer Rate", value=f"{returning_customer_rate:.2f}%")
    with col2:
        st.metric(label="Most Loyal Customer", value=f"{most_loyal_customer_ever} ({most_loyal_customer_count_ever} reviews)")

    st.write("###")
    st.write("#### Overall Mean Ratings")
    # Display overall mean ratings in a row
    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        st.metric(label="Cleanliness", value=f"{overall_mean_cleanliness:.2f}")
    with col4:
        st.metric(label="Sleep Quality", value=f"{overall_mean_sleep_quality:.2f}")
    with col5:
        st.metric(label="Rooms", value=f"{overall_mean_rooms:.2f}")
    with col6:
        st.metric(label="Service", value=f"{overall_mean_service:.2f}")
    with col7:
        st.metric(label="Value", value=f"{overall_mean_value:.2f}")

    st.write("###")

    # Filter data for the past 12 months based on the latest month in the df
    df['date_review'] = pd.to_datetime(df['date_review'])
    latest_month = df['date_review'].max().to_period('M')
    past_12_months_start = (latest_month - 11).to_timestamp()
    last_12_months = df[df['date_review'] >= past_12_months_start]

    # Metrics for the past 12 months
    st.write("## Past 12 Months Metrics")

    # Monthly evolution of positive reviews ratio
    st.write("#### Positive Review Ratio (past 12 months)")
    monthly_positive_ratio = last_12_months[last_12_months['sentiment'] == 1].groupby(last_12_months['date_review'].dt.to_period('M')).size() / last_12_months.groupby(last_12_months['date_review'].dt.to_period('M')).size()
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    monthly_positive_ratio.plot(kind='line', ax=ax1, color='black')
    ax1.set_xlabel("Month", fontsize=25)
    ax1.set_ylabel("Positive Reviews Ratio", fontsize=25)
    ax1.tick_params(axis='y', which='major', labelsize=25)  # Set font size for y-axis tick labels
    st.pyplot(fig1)

    st.write("#### Number of Reviews (past 12 months)")
    monthly_reviews = last_12_months['date_review'].dt.to_period('M').value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    monthly_reviews.plot(kind='bar', ax=ax2, color='gray')
    ax2.set_xlabel("Month", fontsize=25)
    ax2.set_ylabel("Number of Reviews", fontsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    st.pyplot(fig2)

    st.write("###")

    # Metrics for the latest month
    st.write("## Latest Month Metrics")

    # Filter data for the latest month
    latest_month_reviews = df[df['date_review'].dt.to_period('M') == latest_month]

    # Calculate metrics for the latest month
    latest_month_sentiment_counts = latest_month_reviews['sentiment'].value_counts(normalize=True)
    latest_month_positive_ratio = latest_month_sentiment_counts[1] * 100 if 1 in latest_month_sentiment_counts else 0
    latest_month_mean_rating = latest_month_reviews['mean_rating'].mean()
    latest_month_mean_cleanliness = latest_month_reviews['cleanliness'].mean()
    latest_month_mean_sleep_quality = latest_month_reviews['sleep_quality'].mean()
    latest_month_mean_rooms = latest_month_reviews['rooms'].mean()
    latest_month_mean_service = latest_month_reviews['service'].mean()
    latest_month_mean_value = latest_month_reviews['value'].mean()

    # Calculate % change from general ratings
    def calculate_percentage_change(current, general):
        return ((current - general) / general) * 100

    cleanliness_change = calculate_percentage_change(latest_month_mean_cleanliness, overall_mean_cleanliness)
    sleep_quality_change = calculate_percentage_change(latest_month_mean_sleep_quality, overall_mean_sleep_quality)
    rooms_change = calculate_percentage_change(latest_month_mean_rooms, overall_mean_rooms)
    service_change = calculate_percentage_change(latest_month_mean_service, overall_mean_service)
    value_change = calculate_percentage_change(latest_month_mean_value, overall_mean_value)

    # Calculate % change for the main metrics
    positive_ratio_change = calculate_percentage_change(latest_month_positive_ratio, positive_ratio)
    mean_rating_change = calculate_percentage_change(latest_month_mean_rating, overall_mean_rating)

    # Most loyal customer in the latest month
    most_loyal_customer_latest_month = latest_month_reviews['author'].value_counts().idxmax()
    most_loyal_customer_count_latest_month = latest_month_reviews['author'].value_counts().max()

    # Number of reviews in the latest month
    number_of_reviews_latest_month = latest_month_reviews.shape[0]

    # Calculate the number of reviews in the prior month
    prior_month = (latest_month - 1).to_timestamp()
    prior_month_reviews = df[df['date_review'].dt.to_period('M') == prior_month]
    number_of_reviews_prior_month = prior_month_reviews.shape[0]

    # Calculate % change for the number of reviews compared to the prior month
    if number_of_reviews_prior_month > 0:
        number_of_reviews_change = calculate_percentage_change(number_of_reviews_latest_month, number_of_reviews_prior_month)
    else:
        number_of_reviews_change = None  # Handle division by zero if there are no reviews in the prior month

    # Display latest month metrics side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Positive Reviews Ratio", value=f"{latest_month_positive_ratio:.2f}%", delta=f"{positive_ratio_change:+.2f}%")
    with col2:
        st.metric(label="Mean Rating", value=f"{latest_month_mean_rating:.2f}", delta=f"{mean_rating_change:+.2f}%")
    with col3:
        if number_of_reviews_change is not None:
            st.metric(label="Nº of Reviews", value=f"{number_of_reviews_latest_month}", delta=f"{number_of_reviews_change:+.2f}%")
        else:
            st.metric(label="Nº of Reviews", value=f"{number_of_reviews_latest_month}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Most Loyal Customer", value=f"{most_loyal_customer_latest_month} ({most_loyal_customer_count_latest_month} reviews)")

    st.write("###")
    st.write("#### Latest Month Mean Ratings")
    # Display latest month mean ratings in a row with % change
    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        st.metric(label="Cleanliness", value=f"{latest_month_mean_cleanliness:.2f}", delta=f"{cleanliness_change:+.2f}%")
    with col4:
        st.metric(label="Sleep Quality", value=f"{latest_month_mean_sleep_quality:.2f}", delta=f"{sleep_quality_change:+.2f}%")
    with col5:
        st.metric(label="Rooms", value=f"{latest_month_mean_rooms:.2f}", delta=f"{rooms_change:+.2f}%")
    with col6:
        st.metric(label="Service", value=f"{latest_month_mean_service:.2f}", delta=f"{service_change:+.2f}%")
    with col7:
        st.metric(label="Value", value=f"{latest_month_mean_value:.2f}", delta=f"{value_change:+.2f}%")

    # Generate the report content
    report = io.StringIO()
    report.write(f"Hotel Name: {hotel_name}\n")
    report.write(f"\n## General Metrics\n")
    report.write(f"Positive Reviews Ratio: {positive_ratio:.2f}%\n")
    report.write(f"Mean Rating: {overall_mean_rating:.2f}\n")
    report.write(f"Returning Customer Rate: {returning_customer_rate:.2f}%\n")
    report.write(f"Most Loyal Customer: {most_loyal_customer_ever} ({most_loyal_customer_count_ever} reviews)\n")
    report.write(f"\n### Overall Mean Ratings\n")
    report.write(f"Cleanliness: {overall_mean_cleanliness:.2f}\n")
    report.write(f"Sleep Quality: {overall_mean_sleep_quality:.2f}\n")
    report.write(f"Rooms: {overall_mean_rooms:.2f}\n")
    report.write(f"Service: {overall_mean_service:.2f}\n")
    report.write(f"Value: {overall_mean_value:.2f}\n")

    report.write(f"\n## Past 12 Months Metrics\n")
    report.write(f"Positive Review Ratio (past 12 months):\n")
    for period, ratio in monthly_positive_ratio.items():
        report.write(f"{period}: {ratio:.2f}\n")
    report.write(f"\nNumber of Reviews (past 12 months):\n")
    for period, count in monthly_reviews.items():
        report.write(f"{period}: {count}\n")

    report.write(f"\n## Latest Month Metrics\n")
    report.write(f"Positive Reviews Ratio: {latest_month_positive_ratio:.2f}% ({positive_ratio_change:+.2f}%)\n")
    report.write(f"Mean Rating: {latest_month_mean_rating:.2f} ({mean_rating_change:+.2f}%)\n")
    if number_of_reviews_change is not None:
        report.write(f"Number of Reviews: {number_of_reviews_latest_month} ({number_of_reviews_change:+.2f}%)\n")
    else:
        report.write(f"Number of Reviews: {number_of_reviews_latest_month}\n")
    report.write(f"Most Loyal Customer: {most_loyal_customer_latest_month} ({most_loyal_customer_count_latest_month} reviews)\n")
    report.write(f"\n### Latest Month Mean Ratings\n")
    report.write(f"Cleanliness: {latest_month_mean_cleanliness:.2f} ({cleanliness_change:+.2f}%)\n")
    report.write(f"Sleep Quality: {latest_month_mean_sleep_quality:.2f} ({sleep_quality_change:+.2f}%)\n")
    report.write(f"Rooms: {latest_month_mean_rooms:.2f} ({rooms_change:+.2f}%)\n")
    report.write(f"Service: {latest_month_mean_service:.2f} ({service_change:+.2f}%)\n")
    report.write(f"Value: {latest_month_mean_value:.2f} ({value_change:+.2f}%)\n")

    # Convert report to a downloadable format
    report_content = report.getvalue()
    report_bytes = report_content.encode('utf-8')
    st.download_button(label="Download Full Report", data=report_bytes, file_name="hotel_report.txt", mime="text/plain")

    # Perform ABSA analysis and display results
    st.write("## ABSA Analysis")

    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    df['cleaned_review'] = df['full_text'].apply(preprocess_text)

    # Load the sentiment-analysis pipeline with GPU support
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", tokenizer="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

    # Define aspects
    aspects = ['service', 'cleanliness', 'location', 'food', 'staff', 'room', 'price', 'amenities']

    # Extract aspects and analyze sentiment
    def extract_aspects_and_sentiment(review, aspects, sentiment_analyzer):
        aspect_sentiments = {}
        for aspect in aspects:
            if aspect in review.lower():
                # Ensure the input is properly truncated
                truncated_review = review[:512]
                result = sentiment_analyzer(truncated_review)  # Pass the cleaned text directly
                aspect_sentiments[aspect] = result
        return aspect_sentiments

    df['aspect_sentiments'] = df['cleaned_review'].apply(lambda x: extract_aspects_and_sentiment(x, aspects, sentiment_analyzer))

    # Aggregate sentiment scores
    aspect_sentiment_summary = []

    for _, row in df.iterrows():
        for aspect, sentiment in row['aspect_sentiments'].items():
            aspect_sentiment_summary.append({
                'Aspect': aspect,
                'Sentiment': sentiment[0]['label'],
                'Score': sentiment[0]['score']
            })

    aspect_sentiment_summary_df = pd.DataFrame(aspect_sentiment_summary)

    # Convert sentiments to numerical scores
    def sentiment_to_score(sentiment):
        if sentiment == '1 star':
            return -2
        elif sentiment == '2 stars':
            return -1
        elif sentiment == '3 stars':
            return 0
        elif sentiment == '4 stars':
            return 1
        elif sentiment == '5 stars':
            return 2
        return 0

    aspect_sentiment_summary_df['Sentiment Score'] = aspect_sentiment_summary_df['Sentiment'].apply(sentiment_to_score)

    # Visualize the results
    summary = aspect_sentiment_summary_df.groupby('Aspect')['Sentiment Score'].mean().reset_index()

    # Separate positive and negative sentiments
    aspect_sentiment_summary_df['Sentiment Type'] = aspect_sentiment_summary_df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

    # Plot combined histogram for all aspects with positive and negative sentiments
    plt.figure(figsize=(14, 8))
    sns.histplot(data=aspect_sentiment_summary_df, x='Aspect', hue='Sentiment Type', multiple='dodge', shrink=0.8, palette={'Positive': 'green', 'Negative': 'red'})
    plt.title('Sentiment Score Distribution for All Aspects')
    plt.xlabel('Aspect')
    plt.ylabel('Frequency')
    plt.legend(title='Sentiment Type')
    st.pyplot(plt)

    # Count the most common positive and negative words
    positive_words = []
    negative_words = []
    
    for index, row in df.iterrows():
        for aspect, sentiment in row['aspect_sentiments'].items():
            if sentiment[0]['label'] == '4 stars' or sentiment[0]['label'] == '5 stars':
                positive_words.extend(row['cleaned_review'].split())
            elif sentiment[0]['label'] == '1 star' or sentiment[0]['label'] == '2 stars':
                negative_words.extend(row['cleaned_review'].split())

    positive_word_counts = Counter(positive_words)
    negative_word_counts = Counter(negative_words)

    # Get the most common positive and negative words
    most_common_positive_words = positive_word_counts.most_common(10)
    most_common_negative_words = negative_word_counts.most_common(10)

    # Generate the text report
    report = io.StringIO()
    report.write("## Most Common Positive Words\n")
    for word, count in most_common_positive_words:
        report.write(f"{word}: {count}\n")

    report.write("\n## Most Common Negative Words\n")
    for word, count in most_common_negative_words:
        report.write(f"{word}: {count}\n")


    # Display the most common positive and negative words as a table
    st.write("### Most Common Positive Words")
    positive_words_df = pd.DataFrame(most_common_positive_words, columns=['Word', 'Count'])
    st.table(positive_words_df)

    st.write("### Most Common Negative Words")
    negative_words_df = pd.DataFrame(most_common_negative_words, columns=['Word', 'Count'])
    st.table(negative_words_df)

    # Convert report to a downloadable format
    report_content = report.getvalue()
    report_bytes = report_content.encode('utf-8')
    st.download_button(label="Download ABSA Text Report", data=report_bytes, file_name="absa_report.txt", mime="text/plain")

################################################################################
# Sidebar for navigation
st.sidebar.title('Navigation')
section = st.sidebar.radio('Go to', ['Sentiment Analysis', 'ABSA'])

if section == 'Sentiment Analysis':
    st.write("# Sentiment Analysis for Hotel Reviews")
    st.write('Enter a review to get the sentiment analysis output (Positive or Negative).')

    # Text input for the review
    review = st.text_area('Review', '')

    # Button to analyze the sentiment and get suggested response
    if st.button('Analyze Sentiment'):
        if review:
            # Transform the input review using the TF-IDF vectorizer
            review_vec = vectorizer.transform([review])
            
            # Predict the sentiment using the loaded model
            prediction = model.predict(review_vec)
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
            
            # Display the sentiment
            st.write(f'The sentiment of the review is: **{sentiment}**')
            
            # Get and display the suggested response from ChatGPT
            suggested_response = get_chatgpt_response(review)
            st.write('### Suggested Response')
            st.write(suggested_response)

            # Get and display the summary report for feedback
            summary_response = get_summary_response(review)
            st.write('### Feedback Summary')
            st.write(summary_response)

        else:
            st.write('Please enter a review.')

elif section == 'ABSA':
    st.write('# Select a Hotel to generate a report.')

    # Dropdown to select dataset
    dataset_name = st.selectbox('Select Dataset', ['data.csv', 'data_positive.csv','data_negative.csv' ])

    # Button to load the dataset
    if st.button('Load Dataset'):
        df = load_dataset(dataset_name)

        # Perform calculations and display a reporting dashboard
        plot_summary_statistics(df)
