import streamlit as st
st.set_page_config(page_title="AI News Bias Detector", page_icon="📰")
import requests
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv 

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

@st.cache_resource
def load_bias_model():
    tokenizer = AutoTokenizer.from_pretrained("bucketresearch/politicalBiasBERT")
    model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
    return tokenizer, model

tokenizer, bias_model = load_bias_model()

def fetch_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return [
                f"{a['title']} - {a['description']}"
                for a in resp.json().get("articles", []) if a.get("description")
            ]
        else:
            return [f"Error fetching: HTTP {resp.status_code}"]
    except Exception as e:
        return [f"Error: {e}"]

def detect_bias_simple(text):
    left_kw = ['progressive', 'liberal', 'social justice', 'equality', 'climate action', 'healthcare for all', 'workers rights', 'diversity', 'renewable']
    right_kw = ['conservative', 'traditional', 'free market', 'strong defense', 'law and order', 'fiscal responsibility', 'business', 'patriot']
    tl = text.lower()
    matched_left = [kw for kw in left_kw if kw in tl]
    matched_right = [kw for kw in right_kw if kw in tl]
    l, r = len(matched_left), len(matched_right)
    if l + r == 0:
        return "Neutral", 0.5, matched_left, matched_right
    if l > r:
        return "Left", l / (l + r), matched_left, matched_right
    elif r > l:
        return "Right", r / (l + r), matched_left, matched_right
    else:
        return "Center", 0.5, matched_left, matched_right

def detect_bias(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bias_model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0].tolist()
        label_map = {0: "Left", 1: "Center", 2: "Right"}
        idx = int(torch.argmax(outputs.logits[0]))
        ml_label, ml_score = label_map[idx], probs[idx]
    except Exception:
        st.warning("⚠️ ML model failed.")
        ml_label, ml_score = "Unavailable", 0.0

    simple_label, simple_score, matched_left, matched_right = detect_bias_simple(text)
    return (ml_label, ml_score), (simple_label, simple_score, matched_left, matched_right)

def interpret_sentiment(p, s):
    return (
        "Positive" if p > 0.1 else "Negative" if p < -0.1 else "Neutral",
        "Subjective" if s > 0.5 else "Objective"
    )

st.title("📰 News Bias Detector (BERT + Keyword)")
st.markdown("""
Analyze news articles for *political bias* (Left/Center/Right) and *sentiment*.
Compare machine learning (BERT) and keyword-based detection.
""")

st.subheader("✍ Paste Article")
article = st.text_area("Paste the news article text here:")

if st.button("Analyze Pasted Article"):
    if article.strip():
        blob = TextBlob(article)
        pol, subj = blob.sentiment
        sent, obj = interpret_sentiment(pol, subj)
        st.metric("🧠 Sentiment", sent, f"{pol:.2f}")
        st.metric("📊 Objectivity", obj, f"{subj:.2f}")

        (ml_label, ml_score), (simple_label, simple_score, matched_left, matched_right) = detect_bias(article)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 BERT Model")
            st.metric("Bias", ml_label, f"{ml_score:.2f}")
        with col2:
            st.subheader("🧠 Keyword-Based")
            st.metric("Bias", simple_label, f"{simple_score:.2f}")
            st.caption(f"Matched left keywords: {', '.join(matched_left) or 'None'}")
            st.caption(f"Matched right keywords: {', '.join(matched_right) or 'None'}")
    else:
        st.warning("Please paste some text.")

st.markdown("---")

st.subheader("🌐 Analyze by Topic")
topic = st.text_input("Enter a topic (e.g., elections, climate)")

if st.button("Fetch & Analyze"):
    if topic.strip():
        with st.spinner("Fetching news..."):
            arts = fetch_news(topic)
        if arts and not arts[0].startswith("Error"):
            st.success(f"Found {len(arts)} articles.")
        else:
            st.error(arts[0])
            arts = []

        for i, txt in enumerate(arts, start=1):
            with st.expander(f"Article {i}", expanded=True):
                st.write(txt)
                blob = TextBlob(txt)
                pol, subj = blob.sentiment
                sent, obj = interpret_sentiment(pol, subj)
                c1, c2 = st.columns(2)
                c1.metric("🧠 Sentiment", sent, f"{pol:.2f}")
                c2.metric("📊 Objectivity", obj, f"{subj:.2f}")

                (ml_label, ml_score), (simple_label, simple_score, matched_left, matched_right) = detect_bias(txt)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🤖 BERT Model")
                    st.metric("Bias", ml_label, f"{ml_score:.2f}")
                with col2:
                    st.subheader("🧠 Keyword-Based")
                    st.metric("Bias", simple_label, f"{simple_score:.2f}")
                    st.caption(f"Matched left keywords: {', '.join(matched_left) or 'None'}")
                    st.caption(f"Matched right keywords: {', '.join(matched_right) or 'None'}")
                st.markdown("---")
    else:
        st.warning("Enter a topic first.")