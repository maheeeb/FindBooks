import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("books.csv")

# Load BERT model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Recommend books
def recommend(user_input, df, model, top_k=3):
    embeddings = model.encode(df["description"])
    query_embedding = model.encode([user_input])
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

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