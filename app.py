import pickle
import streamlit as st
# Load the trained model and vectorizer
with open("model.pkl", "rb") as file:
    classifier = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Streamlit app
st.header("Sentiment analysis using Twitter data")
d = st.text_input("Enter your tweet")
b = st.button("Submit")
if b :
    if d.strip() == "":
        st.error("Enter some text")
    # Transform the single string by wrapping it in a list

    new_data_counts = vectorizer.transform([d])

    # Make prediction on the transformed data
    new_prediction = classifier.predict(new_data_counts)
    if new_prediction[0]==4:
        st.success("The tweet is positive")
    elif new_prediction==0:
        st.error("This tweet is negative")
    else:
        st.warning("This is neutral")
