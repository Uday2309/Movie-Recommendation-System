# preprocess.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 — Load dataset
movies = pd.read_csv("dataset.csv")  # Make sure dataset.csv is in the same folder
print("✅ Dataset Loaded Successfully")
print(movies.info())

# Step 2 — Handle missing values
movies['genre'] = movies['genre'].fillna('')
movies['overview'] = movies['overview'].fillna('')
movies['original_language'] = movies['original_language'].fillna('unknown')
movies['release_date'] = movies['release_date'].fillna('0000')

# Step 3 — Combine text features into one
movies['tags'] = movies['genre'] + ' ' + movies['overview']

# Step 4 — Select relevant columns
new_df = movies[['title', 'tags', 'vote_average', 'original_language', 'release_date']]
print("\n🎬 Sample Data:")
print(new_df.head(3))

# Step 5 — Convert text to numeric vectors using TF-IDF
print("\n🔍 Converting text into TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Step 6 — Compute similarity matrix
print("\n🧮 Calculating cosine similarity matrix...")
similarity = cosine_similarity(vectors)

# Step 7 — Normalize ratings (0 to 1 scale)
movies['vote_average_norm'] = (
    (movies['vote_average'] - movies['vote_average'].min()) /
    (movies['vote_average'].max() - movies['vote_average'].min())
)

# Step 8 — Recommendation function with language & year filter
def recommend(movie, lang=None, year=None):
    movie = movie.lower().strip()
    if movie not in new_df['title'].str.lower().values:
        print("❌ Movie not found in dataset.")
        return

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[index]

    # Weighted similarity = 70% content + 30% normalized rating
    weighted_similarity = (0.7 * distances) + (0.3 * movies['vote_average_norm'])

    # Sort by highest similarity score
    movie_list = sorted(list(enumerate(weighted_similarity)), reverse=True, key=lambda x: x[1])[1:]  # Skip same movie

    # Collect top recommendations
    recommendations = []
    count = 0
    for i in movie_list:
        rec_movie = new_df.iloc[i[0]]
        release_year = rec_movie['release_date'][:4] if isinstance(rec_movie['release_date'], str) else 'N/A'

        # Apply optional filters
        if lang and rec_movie['original_language'].lower() != lang.lower():
            continue
        if year and release_year != str(year):
            continue

        recommendations.append({
            'Title': rec_movie['title'],
            'Rating': movies.iloc[i[0]]['vote_average'],
            'Year': release_year,
            'Language': rec_movie['original_language']
        })
        count += 1
        if count == 5:
            break

    # Show recommendations in table
    if recommendations:
        df_recommend = pd.DataFrame(recommendations)
        print("\n🎥 Top Recommended Movies:")
        print(df_recommend.to_string(index=False))
    else:
        print("\n⚠️ No movies found with the given filters.")

# Step 9 — Example calls
if __name__ == "__main__":
    print("\n✅ Model Ready! Try recommendations below 👇")
    recommend("The Shawshank Redemption")
    recommend("The Godfather", lang="en")
    recommend("Dilwale Dulhania Le Jayenge", lang="hi")
    recommend("Avengers: Endgame", year=2019)
