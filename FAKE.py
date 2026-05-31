import streamlit as st
import pickle

model = pickle.load(open("truthlens_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection")

news = st.text_area("Enter news text")

if st.button("Predict"):
    if news.strip():
        vec = vectorizer.transform([news])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.write("Raw prediction:", pred)

        if pred == 1:
            st.success(f"✅ REAL NEWS ({prob[1]*100:.2f}% confidence)")
        else:
            st.error(f"🚨 FAKE NEWS ({prob[0]*100:.2f}% confidence)")
    else:
        st.warning("Please enter some news text.")