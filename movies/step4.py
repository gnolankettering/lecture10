# generate embeddings for movie plots and store in cache (pickle file)
from openai import OpenAI
import config
import pandas as pd
import pickle
from nomic import atlas


from tenacity import retry, wait_random_exponential, stop_after_attempt

client = OpenAI(api_key=config.OPENAI_API_KEY)

# read in the movies into a data frame
dataset_path = "./wiki_movie_plots_deduped.csv"
df = pd.read_csv(dataset_path)

# filter on American movies, sort by release year, and take the top 5000
movies = df[df["Origin/Ethnicity"] == "American"].sort_values(by="Release Year", ascending=False).head(5000)


# Extract the movie plots into a list - this is what we will use to generate embeddings
movie_plots = movies["Plot"].values

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

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

# add the title and genre to the embeddings
data = movies[["Title", "Genre"]].to_dict("records")
# upload to atlas website
dataset = atlas.map_data(data=data, embeddings=np.array(plot_embeddings))