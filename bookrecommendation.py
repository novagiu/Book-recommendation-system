from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import re
import os


#reading a csv files and checking if a csv is available at a given address
csv_path = "C:/Users/Xiaomi/Downloads/archive (12)/books_data.csv"

if os.path.exists(csv_path):
    books_df = pd.read_csv(csv_path, header=0)
    print('Dataset of our books:')
    print(books_df.head(10))
else:(f"File '{csv_path}' not found. Please check the path.")

csv_path="C:/Users/Xiaomi/Downloads/archive (12)/Books_rating.csv"

if os.path.exists(csv_path):
    books_review_df=pd.read_csv(csv_path, header=0)
    print('Dataset of the rating of out books:')
    print(books_review_df.head(10))
else:(f"File '{csv_path}' not found. Please check the path.")

csv_path = "C:/Users/Xiaomi/Downloads/archive (11)/training.1600000.processed.noemoticon.csv"

if os.path.exists(csv_path):
    twitter_df = pd.read_csv(csv_path, header=0, encoding='ISO-8859-1')
    print('Dataset of tweets of our users:')
    print(twitter_df.head(10))
else:(f"File '{csv_path}' not found. Please check the path.")

#renaming the columns of the twitter DataFrame
twitter_df.rename(columns={
    "0": "sentiment",
    "1467810369": "id",
    "Mon Apr 06 22:19:45 PDT 2009": "date",
    "NO_QUERY": "flag",
    "_TheSpecialOne_": "user",
    "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D": "tweet"
}, inplace=True)
print(twitter_df.head(10))
print('Current column names:')
print(twitter_df.columns)

#check for missing values in the twitter DataFrame
print('Checking for missing values in a dataset.')
missing_values = twitter_df.isnull().sum()
print(missing_values)

twitter_df = twitter_df.drop('flag', axis=1)

#exclude short tweets with less than 70 characters
twitter_df = twitter_df[twitter_df['tweet'].str.len() >= 70]

#group by user and count the number of tweets
user_tweet_counts = twitter_df['user'].value_counts()

#exclude users with less than 100 tweets
valid_users = user_tweet_counts[user_tweet_counts >= 100].index
twitter_df = twitter_df[twitter_df['user'].isin(valid_users)]

print('Dataset after cleaning:')
print(twitter_df)

#text clearing function
def data_cleaning(text):
    if isinstance(text, float) and pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]','',text)
    return text

books_review_df['review/text'] = books_review_df['review/text'].apply(data_cleaning)
books_review_counts = books_review_df['Title'].value_counts()

#exclude books with less than 200 reviews
valid_books = books_review_counts[books_review_counts >= 200].index
books_review_df = books_review_df[books_review_df['Title'].isin(valid_books)]

books_review_df = books_review_df.drop('Price', axis=1)
print('Books Review Dataset after cleaning:')
print(books_review_df.head(10))

#clean the 'sentiment' column and 'tweet' column 
cleanedtweets_df = twitter_df.copy()
cleanedtweets_df['Sentiment'] = cleanedtweets_df['sentiment'].astype(str).str.strip()
cleanedtweets_df['Cleaned_Text'] = cleanedtweets_df['tweet'].apply(data_cleaning)
cleanedtweets_df['tweet'] = cleanedtweets_df['Cleaned_Text']
cleanedtweets_df = cleanedtweets_df.drop(columns=['Cleaned_Text'])
print(cleanedtweets_df.head(10))

#apply the data cleaning function to the 'review/text' column
books_review_df['Cleaned_review'] = books_review_df['review/text'].apply(data_cleaning)
cleanedreview_df = books_review_df[['Cleaned_review']]
review=cleanedreview_df['Cleaned_review']
print(books_review_df.head(10))
 
#loading the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2') 

#function for preprocessing text
def preprocess_text(text):
    sentences = text.split('. ') 
    return sentences

#function to get proposal vectors
def get_sentence_vectors(sentences):
    sentence_vectors = model.encode(sentences)  
    return sentence_vectors

#function for averaging proposal vectors
def get_content_vector(sentence_vectors):
    content_vector = np.mean(sentence_vectors, axis=0)  
    return content_vector

#function to calculate the content vector
def calculate_content_vector(text):
    sentences = preprocess_text(text)  
    sentence_vectors = get_sentence_vectors(sentences)  
    content_vector = get_content_vector(sentence_vectors) 
    return content_vector

#sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

#function to calculate sentiment
def calculate_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores

#calculate content vector and sentiment for each user's tweets
user_vectors = {}
user_sentiments = {}
for user in valid_users:
    user_tweets = cleanedtweets_df[cleanedtweets_df['user'] == user]['tweet'].tolist()
    user_text = ' '.join(user_tweets)
    user_vector = calculate_content_vector(user_text)
    user_sentiment = calculate_sentiment(user_text)
    user_vectors[user] = user_vector
    user_sentiments[user] = user_sentiment

#calculate content vector for each book's reviews
book_vectors = {}
for book in valid_books:
    book_reviews = books_review_df[books_review_df['Title'] == book]['review/text'].tolist()
    book_text = ' '.join(book_reviews)
    book_vector = calculate_content_vector(book_text)
    book_vectors[book] = book_vector

#calculate cosine similarity between each user and each book
recommendations = {}
for user, user_vector in user_vectors.items():
    similarity_scores = {}
    for book, book_vector in book_vectors.items():
        similarity = cosine_similarity([user_vector], [book_vector])[0][0]
        similarity_scores[book] = similarity
    sorted_books = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations[user] = sorted_books[:5] 

#display recommendations for 5 users
for user, recs in list(recommendations.items())[:5]:
    print(f"Recommendations for user {user}:")
    for book, score in recs:
        print(f"Book: {book}, Similarity Score: {score:.2f}")
    print("")