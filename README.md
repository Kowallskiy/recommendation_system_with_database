# Movies Recommendation System

## Introduction
This project explores various techniques to build and compare movies recommendation systems, including Content-based Filtering, Collaborative Filtering, and a Hybrid approach. Additionally, I implemented advanced methods using Neural Collaborative Filtering (NCF) with PyTorch to enhance recommendation quality.

## Features
- **Content-based Filtering**: **Recommends** movies by comparing genre similarities.
- **Collaborative Filtering**: Utilizes user-movie interactions to find and recommend similar movies.
- **Hybrid Filtering**: Combines Content-based and Collaborative Filtering for improved recommendations.
- **Neural Collaborative Filtering (NCF)**: Implements a sophisticated model using PyTorch that leverages user-item interaction data.

## Content-based Filtering

```Python
def recommend_movies_based_on_genre(movie_title, num_recommendations=5):
    try:
        target_movies_genres = movies_df[movies_df['title'] == movie_title]['genres'].iloc[0].split('|')

        movies_df['similarity_genre'] = movies_df['genres'].apply(lambda x: len(set(x.split('|')).intersection(set(target_movies_genres))))

        recommendations = movies_df.sort_values(by='similarity_genre', ascending=False)[['movie_id', 'title', 'similarity_genre']]

        # Exclude the movie itself rom the recommendations
        recommendations = recommendations[recommendations['title'] != movie_title]

        return recommendations.head(num_recommendations)

    except IndexError:
        return pd.DataFrame()
```

Recommendations for the movie Fight Club:

```
      movie_id                                 title  similarity_genre
758        996              Last Man Standing (1996)                 4
4752      7076                        Bullitt (1968)                 4
5895     33437      Unleashed (Danny the Dog) (2005)                 4
2788      3729                          Shaft (1971)                 4
4843      7235  Ichi the Killer (Koroshiya 1) (2001)                 4
```

## Collaborative Filtering

```Python
def recommend_movies_cosine_similarity(movie_title, num_recommendations=5):
    if movie_title not in movies_df['title'].values:
        return pd.DataFrame()
    
    movie_id = movies_df[movies_df['title'] == movie_title]['movie_id'].iloc[0]

    sim_scores = cosine_sim_df[movie_id].sort_values(ascending=False)

    top_movie_ids = sim_scores.iloc[1:num_recommendations+1].index

    recommendations = pd.DataFrame({
        'movie_id': top_movie_ids,
        'similarity_rating': sim_scores.iloc[1:num_recommendations+1].values
    })

    recommendations = recommendations.merge(movies_df[['movie_id', 'title']], on='movie_id')

    return recommendations
```

Recommendations for the movie Fight Club:

```
   movie_id  similarity_rating                                              title
0      2571           0.713937                                 Matrix, The (1999)
1      4226           0.669593                                     Memento (2000)
2      2329           0.649054                          American History X (1998)
3      6874           0.639738                           Kill Bill: Vol. 1 (2003)
4      4993           0.635744  Lord of the Rings: The Fellowship of the Ring,...
```

## Hybrid Filtering

```Python
def hybrid_recommendation(movie_title, num_recommendations=5):
    genre_recommendations = recommend_movies_based_on_genre(movie_title, num_recommendations * 20)
    rating_recommendations = recommend_movies_cosine_similarity(movie_title, num_recommendations * 20)

    hybrid_recommendations = pd.merge(genre_recommendations, rating_recommendations, on='movie_id', how='inner')

    hybrid_recommendations['hybrid_similarity'] = (hybrid_recommendations['similarity_genre'] + hybrid_recommendations['similarity_rating']) / 2

    final_recommendations = hybrid_recommendations.sort_values(by='hybrid_similarity', ascending=False).head(num_recommendations)

    return final_recommendations
```

Recommendations for the movie Fight Club:

|movie_id|title             |hybrid_similarity|
|--------|------------------|-----------------|
|7913209 |Inception (2010)  |2.3077           |
|29378   |The Professional  |2.2639           |
|601664  |City of God (2002)|2.2501           |
|103637  |Die Hard (1988)   |1.7348           |

## NCF

```Python
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size, mlp_layers):
        super(NCF, self).__init__()
        self.user_embed_gmf = nn.Embedding(num_users, embed_size)
        self.item_embed_gmf = nn.Embedding(num_items, embed_size)
        self.user_embed_mlp = nn.Embedding(num_users, embed_size)
        self.item_embed_mlp = nn.Embedding(num_items, embed_size)

        MLP_modules = []
        input_size = embed_size * 2
        for mlp_layer in mlp_layers:
            MLP_modules.append(nn.Linear(input_size, mlp_layer))
            MLP_modules.append(nn.ReLU())
            input_size = mlp_layer
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(embed_size + mlp_layers[-1], 1)

    def forward(self, user_indices, item_indices):
        user_embed_gmf = self.user_embed_gmf(user_indices)
        item_embed_gmf = self.item_embed_gmf(item_indices)
        gmf_out = user_embed_gmf * item_embed_gmf

        user_embed_mlp = self.user_embed_mlp(user_indices)
        item_embed_mlp = self.item_embed_mlp(item_indices)
        mlp_out = torch.cat((user_embed_mlp, item_embed_mlp), -1)
        mlp_out = self.MLP_layers(mlp_out)
        
        concat = torch.cat((gmf_out, mlp_out), -1)
        prediction = self.predict_layer(concat)
        return prediction.squeeze()
```
