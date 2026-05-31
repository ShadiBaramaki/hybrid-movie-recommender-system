import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix


@st.cache_data
def load_data():
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    return movies, ratings


@st.cache_data
def build_cb_matrix(movies):
    cv = CountVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None)
    genre_matrix = cv.fit_transform(movies["genres"])
    return genre_matrix


@st.cache_data
def build_user_movie_sparse(ratings):
    user_movie = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)
    movie_ids = user_movie.columns.tolist()
    sparse = csr_matrix(user_movie.values.T)  # shape: (movies, users)
    return sparse, movie_ids


def hybrid_recommend(movie_title, movies, sparse_matrix, movie_ids, genre_matrix, top_n=5):
    idx_list = movies[movies["title"] == movie_title].index
    if len(idx_list) == 0:
        return pd.DataFrame()

    idx = idx_list[0]
    movie_id = movies.loc[idx, "movieId"]

    if movie_id not in movie_ids:
        return pd.DataFrame()

    movie_pos = movie_ids.index(movie_id)

    # CF: فقط یه ردیف حساب کن نه کل ماتریس
    movie_vec = sparse_matrix[movie_pos]
    cf_scores = cosine_similarity(movie_vec, sparse_matrix).flatten()
    cf_series = pd.Series(cf_scores, index=movie_ids)

    # CB: فقط یه ردیف
    cb_scores_arr = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
    cb_series = pd.Series(cb_scores_arr, index=movies["movieId"].tolist())

    common_ids = cf_series.index.intersection(cb_series.index)
    hybrid_scores = (cf_series[common_ids] + cb_series[common_ids]) / 2
    hybrid_scores = hybrid_scores.drop(movie_id, errors="ignore")

    top_ids = hybrid_scores.sort_values(ascending=False).head(top_n).index
    return movies[movies["movieId"].isin(top_ids)][["title", "genres"]]


# UI
st.title("🎬 Movie Recommender System")
st.markdown("Hybrid Recommendation (Collaborative + Content-Based Filtering)")

with st.spinner("Loading data..."):
    movies, ratings = load_data()
    genre_matrix = build_cb_matrix(movies)
    sparse_matrix, movie_ids = build_user_movie_sparse(ratings)

st.success("Model ready!")

movie_list = sorted(movies["title"].tolist())

selected_movie = st.selectbox(
    "🔍 Search & Select Movie",
    movie_list,
    index=None,
    placeholder="Type movie name..."
)

top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

if st.button("🎯 Recommend Movies"):
    if selected_movie is None:
        st.warning("Please select a movie first.")
    else:
        with st.spinner("Finding recommendations..."):
            results = hybrid_recommend(
                selected_movie, movies, sparse_matrix, movie_ids, genre_matrix, top_n
            )

        if results.empty:
            st.error("Movie not found in recommendation database.")
        else:
            st.success(f"Top {top_n} recommendations for: **{selected_movie}**")
            for i, row in results.iterrows():
                st.markdown(f"### 🎬 {row['title']}")
                st.markdown(f"🎭 Genre: `{row['genres']}`")
                st.divider()
