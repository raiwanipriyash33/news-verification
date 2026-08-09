import streamlit as st
import pandas as pd
import re
import string
import joblib

# Cache the models so they load only once and save memory
@st.cache_resource
def load_assets():
    vectorization = joblib.load('vectorizer.pkl')
    LR = joblib.load('lr_model.pkl')
    DT = joblib.load('dt_model.pkl')
    return vectorization, LR, DT

try:
    vectorization, LR, DT = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_lable(n):
    if n == 0:
        return "Fake News 🚨"
    elif n == 1:
        return "Not A Fake News ✅"

st.set_page_config(page_title="News Verification Terminal", layout="wide")

st.title("News Reliability Verification Terminal")
st.write("Input the raw text of the article below to run it against the detection algorithms.")

news_input = st.text_area("Article Text:", height=250)

if st.button("Run Analysis"):
    if not news_input:
        st.warning("Please enter some text to analyze.")
    else:
        clean_text = wordopt(news_input)
        vectorized_text = vectorization.transform([clean_text])
        
        pred_LR = LR.predict(vectorized_text)[0]
        pred_DT = DT.predict(vectorized_text)[0]
        
        st.subheader("Analysis Results:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Logistic Regression**\n\n{output_lable(pred_LR)}")
        with col2:
            st.info(f"**Decision Tree**\n\n{output_lable(pred_DT)}")lable(pred_DT)}")
