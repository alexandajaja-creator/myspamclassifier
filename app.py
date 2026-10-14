import streamlit as st
import joblib

# To configure the app page
st.set_page_config(
    page_title="AJs Spam Classifier",
    page_icon=":robot_face:",
    layout="centered"
)

# To load my saved model
@st.cache_resource
def load_artifacts():
    model = joblib.load("AJs_spam_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# To give the app a heading
st.title("AJs Spam Classifier")
st.write("Check if your message is spam or legitimate.")

# Message input from the user
message = st.text_area(
    "Enter your message here:",
    placeholder="Example: Congratulations! You have won a prize.....",
    height=150
)

# Prediction Button
if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        message_tfidf = vectorizer.transform([message])
        spam_probability = model.predict_proba(message_tfidf)[0][1]

        #Setting threshold
        threshold = 0.30

        if spam_probability >= threshold:
            st.error("This message is likely a spam.")
        else:
            st.success("This message is legitimate.")

        st.metric(
            "Spam probability",
            f"{spam_probability:.2%}"
        )
st.caption(" AI/ML Project --- predictions are estimates not guranteed.")
