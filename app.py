import streamlit as st
from src.sentiment_dashboard.model import predict

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💬")

st.title("Sentiment Analysis Dashboard")
st.write("Enter a product review below and see the predicted sentiment.")

text = st.text_area("Review text", placeholder="e.g. This coffee tastes amazing, I'll definitely buy again!")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            result = predict(text)
            label = result["label"]
            confidence = result["confidence"]

            if label == "positive":
                st.success(f"Sentiment: **Positive** (confidence: {confidence:.1%})")
            else:
                st.error(f"Sentiment: **Negative** (confidence: {confidence:.1%})")

        except FileNotFoundError as e:
            st.error(str(e))