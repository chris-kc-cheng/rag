import os
import pandas as pd
import streamlit as st
import kagglehub

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("gpreda/bbc-news")
    return pd.read_csv(os.path.join(path, "bbc_news.csv"))