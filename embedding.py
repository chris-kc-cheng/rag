import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel
from utils import load_data

measures = {
    'Cosine Similarity': (cosine_similarity, -1),
    'Euclidean Distance': (euclidean_distances, 1),
    'Dot Product': (linear_kernel, -1),
}


@st.cache_data
def load_model(model_name):
    return SentenceTransformer(model_name)


@st.cache_data
def load_embeddings(model_name):
    return joblib.load("models/" + model_name.split("/")[-1] + ".joblib")


if "words" not in st.session_state:
    st.session_state.words = ["cat", "dog", "apple",
                              "banana", "car", "automobile", "Canada", "Toronto"]

with st.sidebar:
    model_name = st.selectbox(
        "Model", options=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"], index=0)
    measure = st.selectbox("Measure", options=measures, index=0)
    query = st.text_input("Query", value="Retrieval augmented generation")
    k = st.slider("Top", min_value=1, max_value=20, value=10)

model = load_model(model_name)
news_data = load_data()
data = list(news_data["title"] + " " + news_data["description"])

st.title("Semantic Search")

embeddings = load_embeddings(model_name)
query_embeddings = model.encode(query)

sign = measures[measure][1]
similarity = measures[measure][0]([query_embeddings], embeddings)[0]
sim = sign * np.sort(sign * similarity)[:k]
idx = np.argsort(sign * similarity)[:k]

for j, i in enumerate(idx):
    with st.expander(f"**Rank {j + 1}**: {measure} {sim[j]:.2f}", expanded=True):
        st.markdown(f"**{news_data.iloc[i]['title']}**")
        st.text(news_data.iloc[i]["description"])

st.header("Visualization")
is_3d = st.toggle("3D", value=True)


def add_word():
    word = st.session_state.new_word
    if word:
        st.session_state.words.append(word)
        st.session_state.new_word = ""


word = st.text_input("Add a new word/sentence",
                     key="new_word", on_change=add_word)

# Visualization

vectors = model.encode(st.session_state.words)
dimensions = 3 if is_3d else 2
pca = PCA(n_components=dimensions)
reduced = pca.fit_transform(vectors)
df = pd.DataFrame(
    reduced,
    index=st.session_state.words,
    columns=[f"Principal Component {i + 1}" for i in range(dimensions)]
)
df.index.name = "Sentence"
df = df.reset_index()

if is_3d:
    scatter_3d = px.scatter_3d(
        df,
        x="Principal Component 1", y="Principal Component 2", z="Principal Component 3",
        color="Sentence",
        text="Sentence",
        hover_name="Sentence",
    ).update_layout(
        height=800,
        scene=dict(
            xaxis=dict(title="", showticklabels=False),
            yaxis=dict(title="", showticklabels=False),
            zaxis=dict(title="", showticklabels=False),
        ),
        showlegend=False
    )
    st.plotly_chart(scatter_3d, width='stretch')
else:
    scatter_2d = alt.Chart(df).mark_circle(size=100).encode(
        x="Principal Component 1",
        y="Principal Component 2",
        color=alt.Color("Sentence:N", legend=None),
        tooltip=[
            alt.Tooltip("Sentence:N", title="Sentence"),
            alt.Tooltip("Principal Component 1:Q", title="PC1", format=".3f"),
            alt.Tooltip("Principal Component 2:Q", title="PC2", format=".3f")
        ]
    ).interactive()

    labels = alt.Chart(df).mark_text(
        align="left",
        baseline="middle",
        dx=7,
        dy=-7,
        fontSize=12
    ).encode(
        x="Principal Component 1",
        y="Principal Component 2",
        text="Sentence:N"
    )

    st.altair_chart(scatter_2d + labels)

with st.expander("Data"):
    st.dataframe(df)
