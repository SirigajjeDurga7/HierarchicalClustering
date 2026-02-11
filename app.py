import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        color: black;
    }

    html, body, [class*="css"]  {
        color: black !important;
    }

    .stMarkdown, .stText, .stDataFrame, .stTable {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("News Topic Discovery Dashboard")

st.markdown("""
This system uses **Hierarchical Clustering** to automatically group 
similar news articles based on textual similarity.

👉 Discover hidden themes without defining categories upfront.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("all-data.csv", encoding="latin1")
    df.columns = ["sentiment", "text"]
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

text_data = df["text"].astype(str)

st.sidebar.header("Text Vectorization Controls")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features", 100, 2000, 1000
)

remove_stopwords = st.sidebar.checkbox(
    "Use English Stopwords", True
)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

stop_words = "english" if remove_stopwords else None

tfidf = TfidfVectorizer(
    max_features=max_features,
    stop_words=stop_words,
    ngram_range=ngram_range,
    lowercase=True
)

X = tfidf.fit_transform(text_data)

st.sidebar.header("Hierarchical Clustering Controls")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

dendro_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)

if st.button("Generate Dendrogram"):

    st.subheader("Dendrogram Visualization")

    X_subset = X[:dendro_size].toarray()

    fig = plt.figure(figsize=(12, 6))

    Z = sch.linkage(X_subset, method=linkage_method)
    sch.dendrogram(Z, no_labels=True)

    plt.title("Dendrogram")
    plt.xlabel("Articles")
    plt.ylabel("Distance")

    st.pyplot(fig)

    st.info("""
Large vertical gaps indicate strong separation between topics.
Choose cluster count based on these gaps.
""")

st.sidebar.header("Clustering Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2, 10, 4
)

if st.button("Apply Clustering"):

    X_dense = X.toarray()

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )

    labels = model.fit_predict(X_dense)
    df["Cluster"] = labels

    st.subheader("Cluster Distribution")
    st.write(df["Cluster"].value_counts())

    score = silhouette_score(X_dense, labels)

    st.subheader("Silhouette Score")
    st.write("Score:", round(score, 4))

    if score > 0.5:
        st.success("Well-separated clusters.")
    elif score > 0.2:
        st.info("Moderate separation.")
    elif score > 0:
        st.warning("Clusters overlap slightly.")
    else:
        st.error("Poor clustering structure.")

    st.subheader("Cluster Summary")

    feature_names = tfidf.get_feature_names_out()
    summary_data = []

    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_texts = X_dense[cluster_indices]

        mean_tfidf = np.mean(cluster_texts, axis=0)
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        snippet = text_data.iloc[cluster_indices[0]][:200]

        summary_data.append({
            "Cluster ID": cluster,
            "Number of Articles": len(cluster_indices),
            "Top Keywords": ", ".join(top_words),
            "Sample Snippet": snippet
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

    st.subheader("Business Interpretation")

    for row in summary_data:
        st.markdown(f"""
### Cluster {row['Cluster ID']}
Likely represents articles discussing themes related to:
**{row['Top Keywords']}**
""")

    st.info("""
Articles grouped in the same cluster share similar vocabulary and themes. 
These clusters can be used for automatic tagging, recommendations, 
and content organization.
""")
