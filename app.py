import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


movies = pd.read_csv("data/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")


user_movie = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

cf_sim = cosine_similarity(user_movie.T)

cf_df = pd.DataFrame(
    cf_sim,
    index=user_movie.columns,
    columns=user_movie.columns
)

cv = CountVectorizer(tokenizer=lambda x: x.split("|"))
genre_matrix = cv.fit_transform(movies["genres"])
cb_sim = cosine_similarity(genre_matrix)

# Hybrid Function
def hybrid_recommend(movie_title, top_n=5):

    # movie index
    idx_list = movies[movies["title"] == movie_title].index

    if len(idx_list) == 0:
        return pd.DataFrame()

    idx = idx_list[0]

    # movie id
    movie_id = movies.loc[idx, "movieId"]

    # if movie not in CF matrix
    if movie_id not in cf_df.columns:
        return pd.DataFrame()

    # Collaborative scores
    cf_scores = cf_df[movie_id]

    # Content scores
    cb_scores = pd.Series(
        cb_sim[idx],
        index=movies["movieId"]
    )

    common_ids = cf_scores.index.intersection(cb_scores.index)

    hybrid_scores = (
        cf_scores[common_ids] +
        cb_scores[common_ids]
    ) / 2

    top_ids = hybrid_scores.sort_values(ascending=False).iloc[1:top_n+1].index

    return movies[movies["movieId"].isin(top_ids)][["title", "genres"]]


# UI
st.title("Movie Recommender System")

st.markdown(
    "Find similar movies using a Hybrid Recommendation System "
    "(Collaborative + Content-Based Filtering)"
)

movie_list = sorted(movies["title"].tolist())

selected_movie = st.selectbox(
    "🔍 Search & Select Movie",
    movie_list,
    index=None,
    placeholder="Type movie name..."
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🎯 Recommend Movies"):

    if selected_movie is None:
        st.warning("Please select a movie first.")
    else:

        results = hybrid_recommend(
            selected_movie,
            top_n=top_n
        )

        if results.empty:
            st.error("Movie not found.")
        else:

            st.success(
                f"Top {top_n} recommendations for: {selected_movie}"
            )

            for i, row in results.iterrows():

                st.markdown(f"""
                ### 🎬 {row['title']}
                🎭 Genre: `{row['genres']}`
                """)

                st.divider()