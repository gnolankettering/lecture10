# embeddings function
from openai import OpenAI
import config
import pandas as pd
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

res = get_embedding("Hello, world!")
print(res)