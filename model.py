import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
col = ['sentiment', 'id', 'date', 'flag', 'user', 'tweet']
df = pd.read_csv("D:/Projects/Project Sentiment Analysis/train.csv", encoding='latin-1', names=col)
df.drop(['id', 'date', 'flag', 'user'], axis=1, inplace=True)
print(df.info())

df2=pd.read_csv("D:/Projects/Project Sentiment Analysis/test.csv",names=col)
df2.drop(['id', 'date', 'flag', 'user'], axis=1, inplace=True)
print(df2.info())

df = pd.concat([df, df2], ignore_index=True)
print(df.info())
# Function to remove hashtags, mentions, and URLs from a tweet
def data_preprocess(tweet):
    # Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)

    # Remove @ mentions
    tweet = re.sub(r'@\w+', '', tweet)

    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)

    return tweet

# Preprocess the tweets
df['tweet'] = df['tweet'].apply(data_preprocess)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['sentiment'], test_size=0.2, random_state=42)

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_counts)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(classifier, file)
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("completed")
