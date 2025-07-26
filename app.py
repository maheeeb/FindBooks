import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata
import torch
import nltk

def download_nltk_resources():
    nltk.download('stopwords')
    return True

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("books.csv")

def load_embeddings():
    return torch.load('bert_embeddings.pt').numpy()

# Load BERT model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text, 
               lower=True, 
               remove_punct=True, 
               remove_numbers=False, 
               remove_extra_spaces=True,
               remove_stopwords=False,
               stopwords=None):
    """
    Clean text for NLP tasks with customizable options.
    
    Args:
        text (str): Input text to clean
        lower (bool): Convert to lowercase
        remove_punct (bool): Remove punctuation
        remove_numbers (bool): Remove numbers
        remove_extra_spaces (bool): Normalize whitespace
        remove_stopwords (bool): Whether to remove stopwords
        stopwords (set): Custom stopwords set (if None, uses nltk's English)
    
    Returns:
        str: Cleaned text
    """
    # Ensure text is string and handle missing values
    text = str(text) if text is not None else ""
    
    # Normalize unicode (convert accented characters to ASCII)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Lowercase
    if lower:
        text = text.lower()
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords (requires nltk)
    if remove_stopwords:
        if stopwords is None:
            from nltk.corpus import stopwords
            stopwords = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stopwords])
    
    return text

# Recommend books
def recommend(user_input, df, model, top_k=3):
    embeddings = load_embeddings()
    query_embedding = model.encode([user_input])
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

download_nltk_resources()

# Streamlit UI
st.title("📚 Book Recommender")
user_input = st.text_input("Describe a book you like:", "space adventure")

df = load_data()
model = load_model()

if st.button("Recommend"):
    recommendations = recommend(user_input, df, model)
    st.subheader("Top Recommendations:")
    for _, row in recommendations.iterrows():
        st.write(f"**{row['title']}** (Rating: {row['rating']})")
        st.caption(row["description"])
        st.divider()