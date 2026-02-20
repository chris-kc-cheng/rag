# Retrieval-Augmented Generation (RAG)

This repository is for learning and experimenting with Retrieval-Augmented Generation (RAG) for AI applications.

<a href="https://learn-rag.streamlit.app/">
    <img src='https://static.streamlit.io/badges/streamlit_badge_black_white.svg' alt='Play' style='border: none;' />
</a>

## Information Retrieval and Search

- Keyword Search (exact words, sparse vectors)
  - Bag of words: word order is ignored, only presence and frequency matter
  - TF-IDF (Term Frequency Inverse Document Frequency)
  - BM25 (Best Matching 25) - more commonly used
    $$ IDF = \frac{TF \times (k_1 + 1)}{TF + k_1 \times (1 - b + b(\frac{\text{document length}}{\text{average document length}}))} $$
    where
    - $k_1$ is Term Frequency Saturation
    - $b$ is Length Normalization
- Semantic Search (similar meaning, dense vectors)
  - Embedding models map tokens to a location in space, represented by a vector of many dimensions
  - Contrastive training process using positive/negative pairs
  - Measuring vector distance
    - Euclidean distance: shortest distance
    - Cosine similarity: direction, range from -1 to 1
    - Dot product: projection
  - Reciprocal Rank Function: cares only ranks, not scores
- Metadata Filtering
  - Pros: Simple, fast, enforce strict rules
  - Cons: Rigid, not true search
- Hybrid Search

- Evaluation metrics
  - Precision `Relevant retrieved / Total retrieved`, penalizes for **returning irrelevant** documents
  - Recall `Relevant retrieved / Total Relevant`, penalizes for **leaving out relevant** documents
  - Mean Average Precision or MAP@K evaluate "average precision" (not the above Precision) in the first K documents
  - Mean Reciprocal Rank
  - All metrics requires ground truth relevant documents
- Useful Python libraries
  - `bm25s`, resource module not available on Windows
  - `sentence_transformers`

## Vector Databases
- Approximate Nearest Neighbor algorithm (ANN) does not guarantee to find the absolute cloest documents, but is significantly faster than K Nearest Neighbors (KNN)
- Navigable Small World algorithm uses proximity graphs as the fundamental structure 

## LLMs and Text Generation

## References
