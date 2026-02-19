import streamlit as st
import kagglehub
import os
import pandas as pd
import bm25s

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("gpreda/bbc-news")
    return pd.read_csv(os.path.join(path, "bbc_news.csv"))

with st.sidebar:
    query = st.text_input("Query")
    k = st.slider("k", min_value=1, max_value=20, value=3)

news_data = load_data()

corpus = list(news_data["title"] + " " + news_data["description"])
retriever = bm25s.BM25(corpus=corpus)
corpus_tokens = bm25s.tokenize(corpus)

retriever.index(corpus_tokens)

query_tokens = bm25s.tokenize(query)
docs, scores = retriever.retrieve(query_tokens, k=k)
print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")

st.title("BM25 (Best Matching 25)")
st.write("The BBC data set contains ", len(corpus_tokens.ids), "documents and ", len(corpus_tokens.vocab), " tokens.")
st.write(docs[0])
st.write(scores[0])
