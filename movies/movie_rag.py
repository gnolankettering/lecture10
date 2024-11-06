from openai import OpenAI
import config
import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
from typing import List
from scipy import spatial

client = OpenAI(api_key=config.OPENAI_API_KEY)

# Load the movie database
dataset_path = "./wiki_movie_plots_deduped.csv"
df = pd.read_csv(dataset_path)
movies = df[df["Origin/Ethnicity"] == "American"].sort_values(by="Release Year", ascending=False).head(5000)
movie_plots = movies["Plot"].values

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

# Load the embedding cache
embedding_cache_path = "movie_embeddings_cache.pkl"
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}

def embedding_from_string(
    string: str,
    model: str = "text-embedding-ada-002",
    embedding_cache: dict = embedding_cache
) -> List[float]:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR: {string[:50]}...")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine"
) -> List[float]:
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

def find_similar_movies(
    query: str,
    n_results: int = 5,
    by_plot: bool = True
) -> List[dict]:
    # Get embedding for the query
    query_embedding = embedding_from_string(query)
    
    # Get all stored embeddings
    if by_plot:
        stored_strings = movie_plots
    else:
        stored_strings = movies["Title"].values
        
    stored_embeddings = [embedding_from_string(s) for s in stored_strings]
    
    # Calculate distances
    distances = distances_from_embeddings(query_embedding, stored_embeddings)
    
    # Get indices of nearest neighbors
    nearest_indices = np.argsort(distances)[:n_results]
    
    # Prepare results
    results = []
    for idx in nearest_indices:
        results.append({
            "title": movies.iloc[idx]["Title"],
            "year": movies.iloc[idx]["Release Year"],
            "genre": movies.iloc[idx]["Genre"],
            "plot": movies.iloc[idx]["Plot"],
            "similarity_score": 1 - distances[idx]  # Convert distance to similarity score
        })
    
    return results

def search_by_description(description: str, n_results: int = 5) -> None:
    """Search for similar movies based on a text description."""
    results = find_similar_movies(description, n_results, by_plot=True)
    print(f"\nTop {n_results} similar movies to your description:\n")
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie['title']} ({movie['year']}) - {movie['genre']}")
        print(f"   Similarity Score: {movie['similarity_score']:.2%}")
        print(f"   Plot: {movie['plot'][:200]}...")
        print()

def search_by_title(title: str, n_results: int = 5) -> None:
    """Search for similar movies based on a movie title."""
    results = find_similar_movies(title, n_results, by_plot=False)
    print(f"\nTop {n_results} movies similar to '{title}':\n")
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie['title']} ({movie['year']}) - {movie['genre']}")
        print(f"   Similarity Score: {movie['similarity_score']:.2%}")
        print(f"   Plot: {movie['plot'][:200]}...")
        print()

# Example usage:
if __name__ == "__main__":
    # Search by plot description
    # description = "A sci-fi movie about artificial intelligence taking over the world"
    # search_by_description(description)
    
    # # Search by movie title
    # title = "The Matrix"
    # search_by_title(title)

    # Search by describing the type of movie you want
    search_by_description("A romantic comedy about two people who meet in New York")

    # Or search by a movie title you like
    search_by_title("The Dark Knight")