import os
import pandas as pd
import streamlit as st
import weaviate


@st.cache_resource
def get_database_session():
    session = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=os.environ["WEAVIATE_API_KEY"])
    return session


with st.sidebar:
    query = st.text_input("Query", value="Retrieval augmented generation")
    k = st.slider("Top", min_value=1, max_value=20, value=10)

    with st.expander("Hyperparameters", expanded=True):
        alpha = st.slider("Alpha", min_value=0., max_value=1., value=0.05)

st.title("Vector Database")

with get_database_session() as client:
    news_collection = client.collections.use("bbc_collection")

    df = pd.DataFrame({
        "Keyword": pd.Series([o.properties["title"] for o in news_collection.query.bm25(
            query, limit=k).objects]),
        'Semantic': pd.Series([o.properties["title"] for o in news_collection.query.near_text(query, limit=k).objects]),
        "Hybrid": pd.Series([o.properties["title"] for o in news_collection.query.hybrid(
            query, alpha=alpha, limit=k).objects]),
    })
    st.dataframe(df)
