import streamlit as st

pages = {
    "Retrieval": [
        st.Page("bm25.py", title="Keyword Search (BM25)"),
        st.Page("embedding.py", title="Semantic Search (Embedding)"),
    ],
    "Vector Database": [
        st.Page("db.py", title="Weaviate Cloud"),
    ],
}

st.set_page_config(
    page_title="RAG",
    page_icon=":material/find_in_page:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/chris-kc-cheng",
        "About": "https://www.linkedin.com/in/chris-kc-cheng/"
    }
)

pg = st.navigation(pages, position="top")
pg.run()
