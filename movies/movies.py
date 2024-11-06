from openai import OpenAI
import config
import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
from nomic import atlas
from typing import List
from scipy import spatial

client = OpenAI(api_key=config.OPENAI_API_KEY)

# read in the movies into a data frame
dataset_path = "./wiki_movie_plots_deduped.csv"
df = pd.read_csv(dataset_path)

# # make sure it was read in correctly
# print(df.head())

# # count the number of rows
# print("Number of rows:", len(df))

# filter on American movies, sort by release year, and take the top 5000
movies = df[df["Origin/Ethnicity"] == "American"].sort_values(by="Release Year", ascending=False).head(5000)

# # #count the number of rows
# print("Number of rows:", len(movies))

# Extract the movie plots into a list - this is what we will use to generate embeddings
movie_plots = movies["Plot"].values

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

# # res = get_embedding("Hello, world!")
# # print(res)

# set path to embedding cache
embedding_cache_path = "movie_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string, model="text-embedding-ada-002", embedding_cache=embedding_cache
):
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

plot_embeddings = [embedding_from_string(plot, model="text-embedding-ada-002") for plot in movie_plots]
# # print(plot_embeddings[0])
# # print(len(plot_embeddings))

# add the title and genre to the embeddings
data = movies[["Title", "Genre"]].to_dict("records")
# upload to atlas website
dataset = atlas.map_data(data=data, embeddings=np.array(plot_embeddings))

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)

def print_recommendations_from_strings(strings, index_of_source_string, k_nearest_neighbors=3, model="text-embedding-ada-002"):
    # get all our embeddings
    embeddings = [embedding_from_string(string) for string in strings]
    # get embedding for our specific query string
    query_embedding = embeddings[index_of_source_string]
    # get distances between our embedding and all other embeddings (ex-openai utility function)
    distances = distances_from_embeddings(query_embedding, embeddings)
    # get the indices of the k nearest neighbors (ex-openai utility function)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    # print the k nearest neighbors
    query_string = strings[index_of_source_string] # holds the movie plot
    match_count = 0 # holds the number of plot matches
    for i in indices_of_nearest_neighbors:
        if query_string == strings[i]: # skip the query string plot itself
            continue
        if match_count >= k_nearest_neighbors:
            break   # we have found enough matches
        match_count += 1
        print(f"Found {match_count} closest match: ")
        print(f"Distance of {distances[i]}")
        print(strings[i])

print_recommendations_from_strings(movie_plots, 2) # get recommendations for the 3rd movie plot