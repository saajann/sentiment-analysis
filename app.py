import streamlit as st
import joblib
import pandas as pd

# Carica il modello e il vettorizzatore salvati
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def main():
    st.title("Sentiment Analysis App")
    st.write("This is a basic Streamlit app for sentiment analysis using a custom-trained model.")

    user_input = st.text_area("Enter text to analyze sentiment:")
    incorrect_analysis = st.checkbox("Check if the analysis is incorrect")

    if st.button("Analyze"):
        sentiment = analyze_sentiment(user_input)
        sentiment_label = "Positive" if sentiment == 1 else "Negative"
        st.write(f"Sentiment: {sentiment_label}")
    
    if st.button("Save"):
        sentiment = analyze_sentiment(user_input)
        if incorrect_analysis:
            sentiment = 1 - sentiment  # Invert the sentiment
        save_to_dataset(user_input, sentiment)

def analyze_sentiment(text):
    text_vect = vectorizer.transform([text])
    
    sentiment = model.predict(text_vect)[0]
    
    return sentiment

def save_to_dataset(text, sentiment):
    sentiment_label = 'p' if sentiment == 1 else 'n'
    with open('data/sentiment_dataset.csv', 'a') as f:
        f.write(f"\"{text}\",{sentiment_label}\n")
    st.write("Data saved to dataset.")

if __name__ == "__main__":
    main()
