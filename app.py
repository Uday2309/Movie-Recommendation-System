import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 — Load dataset
movies = pd.read_csv("dataset.csv")

print("✅ Dataset Loaded Successfully")
print(movies.info())

# Step 2 — Handle missing values
movies['genre'] = movies['genre'].fillna('')
movies['overview'] = movies['overview'].fillna('')
movies['original_language'] = movies['original_language'].fillna('unknown')
movies['release_date'] = movies['release_date'].fillna('0000')

# Step 3 — Combine text features
movies['tags'] = movies['genre'] + ' ' + movies['overview']

# Step 4 — Select relevant columns
new_df = movies[['title', 'tags', 'vote_average', 'original_language', 'release_date']]
print("\n🎬 Sample Data:")
print(new_df.head(3))

# Step 5 — TF-IDF Vectorization
print("\n🔍 Converting text into TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Step 6 — Cosine similarity
print("\n🧮 Calculating cosine similarity matrix...")
similarity = cosine_similarity(vectors)

# Step 7 — Normalize ratings
movies['vote_average_norm'] = (
    (movies['vote_average'] - movies['vote_average'].min()) /
    (movies['vote_average'].max() - movies['vote_average'].min())
)

# Step 8 — Recommendation function (with chart/table output)
def recommend(movie, lang=None, year=None):
    movie = movie.lower().strip()
    if movie not in new_df['title'].str.lower().values:
        print("❌ Movie not found in dataset.")
        return

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[index]

    # Weighted similarity = 70% content + 30% rating
    weighted_similarity = (0.7 * distances) + (0.3 * movies['vote_average_norm'])

    # Sort by similarity score
    movie_list = sorted(
        list(enumerate(weighted_similarity)),
        reverse=True,
        key=lambda x: x[1]
    )[1:]

    # Store recommendations
    recommendations = []
    for i in movie_list:
        rec_movie = new_df.iloc[i[0]]
        release_year = rec_movie['release_date'][:4] if isinstance(rec_movie['release_date'], str) else 'N/A'

        if lang and rec_movie['original_language'].lower() != lang.lower():
            continue
        if year and release_year != str(year):
            continue

        recommendations.append({
            'Title': rec_movie['title'],
            'Rating': round(movies.iloc[i[0]]['vote_average'], 1),
            'Year': release_year,
            'Language': rec_movie['original_language']
        })

        if len(recommendations) == 5:
            break

    # If no matches found
    if not recommendations:
        print("\n⚠️ No movies found with the given filters.")
        return

    # Convert to DataFrame for better display
    rec_df = pd.DataFrame(recommendations)
    print(f"\n🎥 Top {len(rec_df)} Movies Similar to '{new_df.iloc[index].title}':\n")
    print(rec_df.to_string(index=False))

    # Plot bar chart
    plt.figure(figsize=(8, 4))
    plt.barh(rec_df['Title'], rec_df['Rating'], color='skyblue')
    plt.xlabel('Rating ★')
    plt.ylabel('Movie Title')
    plt.title(f"Top {len(rec_df)} Recommendations for '{new_df.iloc[index].title}'")
    plt.gca().invert_yaxis()  # Highest on top
    plt.tight_layout()
    plt.show()

# Step 9 — Test
print("\n✅ Model Ready! Try recommendations below 👇")

recommend("The Shawshank Redemption")

