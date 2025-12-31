# preprocess.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_prepare_data():
    movies = pd.read_csv("dataset.csv")

    # Handle missing values
    movies['genre'] = movies['genre'].fillna('')
    movies['overview'] = movies['overview'].fillna('')
    movies['original_language'] = movies['original_language'].fillna('unknown')
    movies['release_date'] = movies['release_date'].fillna('0000')

    # Combine features
    movies['tags'] = movies['genre'] + ' ' + movies['overview']

    new_df = movies[['title', 'tags', 'vote_average', 'original_language', 'release_date']]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    vectors = tfidf.fit_transform(new_df['tags'])

    # Similarity
    similarity = cosine_similarity(vectors)

    # Normalize ratings
    movies['vote_average_norm'] = (
        (movies['vote_average'] - movies['vote_average'].min()) /
        (movies['vote_average'].max() - movies['vote_average'].min())
    )

    return movies, new_df, similarity


def recommend(movie, movies, new_df, similarity, lang=None, year=None):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    weighted_similarity = (0.7 * distances) + (0.3 * movies['vote_average_norm'])

    movie_list = sorted(
        list(enumerate(weighted_similarity)),
        reverse=True,
        key=lambda x: x[1]
    )[1:]

    recommendations = []

    for i in movie_list:
        rec_movie = new_df.iloc[i[0]]
        release_year = rec_movie['release_date'][:4]

        if lang and rec_movie['original_language'].lower() != lang.lower():
            continue
        if year and release_year != year:
            continue

        recommendations.append({
            'Title': rec_movie['title'],
            'Rating': round(movies.iloc[i[0]]['vote_average'], 1),
            'Year': release_year,
            'Language': rec_movie['original_language']
        })

        if len(recommendations) == 5:
            break

    return pd.DataFrame(recommendations)

