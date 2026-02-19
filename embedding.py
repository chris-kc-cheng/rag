import streamlit as st
from utils import load_data
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_model(model_name):
    return SentenceTransformer(model_name)

@st.cache_data
def load_embeddings(model_name, data):
    # 20s for 1000
    # 3:04-3:13 to for all ~40k values
    return load_model(model_name).encode(data)

with st.sidebar:
    model_name = st.selectbox("Model", options=["all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"], index=0)
    query = st.text_input("Query", value="Retrieval augmented generation")
    k = st.slider("Top", min_value=1, max_value=20, value=10)

model = load_model(model_name)
news_data = load_data()

st.title("Semantic Search")

embeddings = load_embeddings(model_name, list(news_data["title"] + " " + news_data["description"]))
query_embeddings = model.encode(query)

st.write(query_embeddings.shape)