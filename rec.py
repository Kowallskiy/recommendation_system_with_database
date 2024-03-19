import os
import pandas as pd
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Fetching environment variables
db_name = os.environ.get('DB_NAME')
db_user = os.environ.get('DB_USER')
db_pass = os.environ.get('DB_PASS')
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT')

# Establish a connection to the database
conn = psycopg2.connect(database=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)

print("Database connected successfully")

query = "SELECT movie_id, title, genres FROM movies;"

movies_df = pd.read_sql_query(query, conn)

def recommend_movies_based_on_genre(movie_title):
    try:
        target_movies_genres = movies_df[movies_df['title'] == movie_title]['genres'].iloc[0].split('|')

        movies_df['similarity'] = movies_df['genres'].apply(lambda x: len(set(x.split('|')).intersection(set(target_movies_genres))))

        recommendations = movies_df.sort_values(by='similarity', ascending=False)

        # Exclude the movie itself rom the recommendations
        recommendations = recommendations[recommendations['title'] != movie_title]

        return recommendations[['title', 'genres', 'similarity']].head(5)

    except IndexError:
        return 'Movie not found'
    
recommendations = recommend_movies_based_on_genre('Fight Club (1999)')
print(recommendations)

query = "SELECT movie_id, user_id, rating FROM ratings;"

ratings_df = pd.read_sql_query(query, conn)

user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
# Transopse the martix so rows represent movies
movie_user_matrix = user_item_matrix.transpose()

# Fill NaN values with 0s as cosine similarity does not work with NaN values
movie_user_matrix_filled = movie_user_matrix.fillna(0)

cosine_sim = cosine_similarity(movie_user_matrix_filled)

cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_user_matrix_filled.index, columns=movie_user_matrix_filled.index)

def recommend_movies_cosine_similarity(movie_title, num_recommendations=5):

    pass