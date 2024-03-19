import os
import pandas as pd
import psycopg2

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