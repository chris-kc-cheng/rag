import streamlit as st
from utils import load_data
import pandas as pd
import bm25s

with st.sidebar:
    query = st.text_input("Query", value="Retrieval augmented generation")
    k = st.slider("Top", min_value=1, max_value=20, value=10)

    with st.expander("Hyperparameters", expanded=True):
        k1 = st.slider("k1", min_value=0., max_value=10., value=1.2)
        b = st.slider("b", min_value=0., max_value=10., value=0.8)

news_data = load_data()

corpus = list(news_data["title"] + " " + news_data["description"])
retriever = bm25s.BM25(corpus=corpus, k1=k1, b=b)
corpus_tokens = bm25s.tokenize(corpus)

retriever.index(corpus_tokens)

query_tokens = bm25s.tokenize(query)
docs, scores = retriever.retrieve(query_tokens, k=k)

id_query = {corpus_tokens.vocab[q]: q for q in list(query_tokens.vocab) if q in corpus_tokens.vocab}
id_doc = [corpus.index(d) for d in docs[0]]
count = pd.DataFrame({d: {id_query[t]: corpus_tokens.ids[d].count(t) for t in id_query} for d in id_doc}).T
score = pd.Series(scores[0], index=id_doc, name="score")
news_data = count.join(news_data).join(score)

st.title("BM25 (Best Matching 25)")
st.write("The BBC data set contains ", len(corpus_tokens.ids), "documents and ", len(corpus_tokens.vocab), " tokens.")
st.write("The query", query, " contains ", len(query_tokens.vocab), " tokens:")
st.markdown(" ".join(f"`{token}`" for token in query_tokens.vocab))

for i, (index, row) in enumerate(news_data.iterrows()):
    with st.expander(f"**Rank {i + 1}**: Score {row['score']:.2f}", expanded=True):
        st.markdown(f"**{row['title']}**")
        st.text(row["description"])        
        st.markdown(" ".join([f":{'green' if row[token] > 0 else 'red'}-badge[{token} ({row[token]})]" for token in id_query.values()]))
