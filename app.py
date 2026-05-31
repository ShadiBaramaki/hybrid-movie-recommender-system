import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np


@st.cache_data
def load_data():
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    return movies, ratings


@st.cache_data
def build_cf_matrix(ratings):
    user_movie = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)
    sparse = csr_matrix(user_movie.values.T)
    cf_sim = cosine_similarity(sparse, dense_output=False)
    return pd.DataFrame(
        cf_sim.toarray(),
        index=user_movie.columns,
        columns=user_movie.columns
    )


@st.cache_data
def build_cb_matrix(movies):
    cv = CountVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None)
    genre_matrix = cv.fit_transform(movies["genres"])
    cb_sim = cosine_similarity(genre_matrix)
    return cb_sim


def hybrid_recommend(movie_title, movies, cf_df, cb_sim, top_n=5):
    idx_list = movies[movies["title"] == movie_title].index
    if len(idx_list) == 0:
        return pd.DataFrame()

    idx = idx_list[0]
    movie_id = movies.loc[idx, "movieId"]

    if movie_id not in cf_df.columns:
        return pd.DataFrame()

    cf_scores = cf_df[movie_id]
    cb_scores = pd.Series(cb_sim[idx], index=movies["movieId"])

    common_ids = cf_scores.index.intersection(cb_scores.index)
    hybrid_scores = (cf_scores[common_ids] + cb_scores[common_ids]) / 2

    top_ids = hybrid_scores.sort_values(ascending=False).iloc[1:top_n+1].index
    return movies[movies["movieId"].isin(top_ids)][["title", "genres"]]


# UI
st.title("Movie Recommender System")
st.markdown(
    "Find similar movies using a Hybrid Recommendation System "
    "(Collaborative + Content-Based Filtering)"
)

with st.spinner("Loading data..."):
    movies, ratings = load_data()

with st.spinner("Building recommendation model (first load only)..."):
    cf_df = build_cf_matrix(ratings)
    cb_sim = build_cb_matrix(movies)

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
        results = hybrid_recommend(selected_movie, movies, cf_df, cb_sim, top_n=top_n)

        if results.empty:
            st.error("Movie not found in recommendation database.")
        else:
            st.success(f"Top {top_n} recommendations for: **{selected_movie}**")
            for i, row in results.iterrows():
                st.markdown(f"### 🎬 {row['title']}")
                st.markdown(f"🎭 Genre: `{row['genres']}`")
                st.divider()
