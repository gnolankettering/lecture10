# download the dataset from https://www.kaggle.com/jrobischon/wikipedia-movie-plots
from openai import OpenAI
import config
import pandas as pd

client = OpenAI(api_key=config.OPENAI_API_KEY)

# read in the movies into a data frame
dataset_path = "./wiki_movie_plots_deduped.csv"
df = pd.read_csv(dataset_path)

# make sure it was read in correctly
print(df.head())

# count the number of rows
print("Number of rows:", len(df))

# filter on American movies, sort by release year, and take the top 500
movies = df[df["Origin/Ethnicity"] == "American"].sort_values(by="Release Year", ascending=False).head(5000)

# #count the number of rows
print("Number of rows:", len(movies))

