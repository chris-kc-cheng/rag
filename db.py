import os
import json
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
        alpha = st.slider("Alpha", min_value=0., max_value=10., value=0.25)

with get_database_session() as client:

    movies = client.collections.use("Movie")

    response = movies.query.near_text(
        query=query,
        limit=2
    )

st.title("Vector Database")

for obj in response.objects:
    st.write(json.dumps(obj.properties, indent=2))
