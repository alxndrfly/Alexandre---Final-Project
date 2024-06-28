import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TextClassificationPipeline
import re
import torch

# Load the csv file reviews
reviews = pd.read_csv('datasets/sql_cleaned/reviews.csv')

# Define the text cleaning function
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the cleaning function and create a new DataFrame with the cleaned text
cleaned_reviews = pd.DataFrame()
cleaned_reviews['clean_text'] = reviews['full_text'].apply(clean_text)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained sentiment analysis model
tokenizer = RobertaTokenizer.from_pretrained('siebert/sentiment-roberta-large-english')
model = RobertaForSequenceClassification.from_pretrained('siebert/sentiment-roberta-large-english')

# Move the model to the GPU
model.to(device)

# Create a sentiment analysis pipeline
sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Function to get sentiment label with truncation
def get_sentiment_label(review):
    inputs = tokenizer(review, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id]
    return label

# Apply the sentiment analysis and add the labels to the DataFrame
reviews['sentiment'] = cleaned_reviews['clean_text'].apply(get_sentiment_label)

# Drop columns to keep only the unprocessed text and the sentiment analysis
reviews.drop(columns=['author', 'date_stayed', 'date_review', 'via_mobile', 'service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms', 'mean_rating'], inplace=True)

# Save the DataFrame as a csv with sentiments added
reviews.to_csv('datasets/model_data/tagged_reviews.csv', index=False)

# We now have tagged data!