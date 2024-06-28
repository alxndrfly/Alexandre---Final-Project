import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load your dataset
data = pd.read_csv('datasets/absa/data.csv')
#data = data[:400]  # Limit to 400 reviews for testing

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

reviews = pd.DataFrame()
reviews['cleaned_review'] = data['full_text'].apply(preprocess_text)

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

reviews['aspect_sentiments'] = reviews['cleaned_review'].apply(lambda x: extract_aspects_and_sentiment(x, aspects, sentiment_analyzer))

# Aggregate sentiment scores
aspect_sentiment_summary = []

for _, row in reviews.iterrows():
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

plt.figure(figsize=(12, 8))
sns.barplot(x='Aspect', y='Sentiment Score', data=summary)
plt.title('Average Sentiment Score per Aspect')
plt.xlabel('Aspect')
plt.ylabel('Average Sentiment Score')
plt.show()

# Plot histograms for each aspect
for aspect in aspects:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=aspect_sentiment_summary_df[aspect_sentiment_summary_df['Aspect'] == aspect], x='Sentiment Score', bins=10)
    plt.title(f'Sentiment Score Distribution for {aspect.capitalize()}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()
# Separate positive and negative sentiments
aspect_sentiment_summary_df['Sentiment Type'] = aspect_sentiment_summary_df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

# Plot combined histogram for all aspects with positive and negative sentiments
plt.figure(figsize=(14, 8))
sns.histplot(data=aspect_sentiment_summary_df, x='Aspect', hue='Sentiment Type', multiple='dodge', shrink=0.8)
plt.title('Sentiment Score Distribution for All Aspects')
plt.xlabel('Aspect')
plt.ylabel('Frequency')
plt.legend(title='Sentiment Type')
plt.show()