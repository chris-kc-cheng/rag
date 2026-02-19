import streamlit as st
import kagglehub
import os
import pandas as pd
import bm25s

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("gpreda/bbc-news")
    return pd.read_csv(os.path.join(path, "bbc_news.csv"))