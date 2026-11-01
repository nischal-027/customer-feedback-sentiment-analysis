
import streamlit as st
import joblib
from pathlib import Path

from preprocessing import preprocess_review


# --------------------------------
# 1. Load trained model and vectorizer
# --------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, tfidf = load_model()


# --------------------------------
# 2. Streamlit page configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)


# --------------------------------
# 3. Application interface
# --------------------------------

st.title("📊 Customer Feedback Sentiment Analysis")

st.write(
    "Enter a customer review below to analyze "
    "whether the feedback is Positive or Negative."
)

st.divider()

review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: The product quality is excellent and delivery was very fast!",
    height=150
)


# --------------------------------
# 4. Sentiment prediction
# --------------------------------

if st.button("Predict Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a review before predicting.")

    else:
        # Preprocess the input review
        processed_review = preprocess_review(review)

        # Convert text into TF-IDF features
        review_vector = tfidf.transform([processed_review])

        # Predict sentiment
        prediction = model.predict(review_vector)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(review_vector)[0]

        confidence = dict(zip(model.classes_, probabilities))

        # Display results
        st.divider()
        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("☹️ Negative Sentiment")

        st.write("**Prediction confidence:**")

        st.write(f"Positive: {confidence.get('Positive', 0) * 100:.2f}%")
        st.progress(float(confidence.get('Positive', 0)))

        st.write(f"Negative: {confidence.get('Negative', 0) * 100:.2f}%")
        st.progress(float(confidence.get('Negative', 0)))

        with st.expander("View preprocessed review"):
            st.write(processed_review)
