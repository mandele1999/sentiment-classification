import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Streamlit App UI
st.set_page_config(page_title="Sentiment Classifier", page_icon="🎭")
st.title("🎭 Customer Feedback Sentiment Classifier")
st.markdown(
    """
    This app uses a **Logistic Regression** model trained on customer feedback to predict whether a sentence expresses a **positive** or **negative** sentiment.

    - Enter a sentence or customer review below.
    - Click **Predict** to see the result.
    """
)

# User input
user_input = st.text_area("𓂃🪶 Enter customer feedback here:")

# Predict button
if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some feedback to classify.")
    else:
        # Transform and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]

        sentiment_label = "Positive" if prediction == 1 else "Negative"
        sentiment_color = "🟢" if prediction == 1 else "🔴"
        confidence = round(np.max(prediction_proba) * 100, 2)

        # Display results
        st.markdown(f"### {sentiment_color} Sentiment: **{sentiment_label}**")
        st.markdown(f"**Model Confidence:** {confidence}%")

        with st.expander("ℹ️ Prediction Details"):
            st.json({
                "Predicted Label": sentiment_label,
                "Confidence (Positive)": f"{round(prediction_proba[1]*100, 2)}%",
                "Confidence (Negative)": f"{round(prediction_proba[0]*100, 2)}%",
                "Model": "Logistic Regression"
            })

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for NLP Portfolio Projects")

