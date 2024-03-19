import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import os
import psycopg2
import torch.nn as nn
import torch.nn.functional as F

db_name = os.environ.get('DB_NAME')
db_user = os.environ.get('DB_USER')
db_pass = os.environ.get('DB_PASS')
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT')

# Establish a connection to the database
conn = psycopg2.connect(database=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)

print("Database connected successfully")

query = "SELECT movie_id, user_id, rating FROM ratings;"

ratings_df = pd.read_sql_query(query, conn)

user_ids = ratings_df['user_id'].unique().tolist()
movie_ids = ratings_df['movie_id'].unique().tolist()

ratings_df['user_id'] = ratings_df['user_id'].apply(lambda x: user_ids.index(x))
ratings_df['movie_id'] = ratings_df['movie_id'].apply(lambda x: movie_ids.index(x))

train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=2)

class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.movies[idx], dtype=torch.long), torch.tensor(self.ratings[idx], dtype=torch.long)
    
train_dataset = MovieDataset(train_data['user_id'].values, train_data['movie_id'].values, train_data['rating'].values)
test_dataset = MovieDataset(test_data['user_id'].values, test_data['movie_id'].values, test_data['rating'].values)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

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
    
num_users = len(user_ids)
num_items = len(movie_ids)
embed_size = 8
mlp_layers = [64, 32, 16, 8]

# Instantiate the model
model = NCF(num_users=num_users, num_items=num_items, embed_size=embed_size, mlp_layers=mlp_layers)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 3

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for user_ids, item_ids, ratings in train_dataloader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.float().to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(user_ids, item_ids)
        
        # Calculate loss
        loss = criterion(outputs, ratings)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

# Evaluation mode
model.eval()
test_loss = 0.0
with torch.no_grad():
    for user_ids, item_ids, ratings in test_dataloader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.float().to(device)
        
        # Forward pass
        outputs = model(user_ids, item_ids)
        
        # Calculate loss
        loss = criterion(outputs, ratings)
        test_loss += loss.item()

# Print average test loss
print(f"Test Loss: {test_loss / len(test_dataloader)}")
