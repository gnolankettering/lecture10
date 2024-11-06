# Dependencies
from datetime import date, timedelta  # date handling for fetching recent news
from IPython import display  # for pretty printing
import json  # for parsing the JSON api responses and model outputs
from numpy import dot  # for cosine similarity
from openai import OpenAI
import requests  # for making the API requests
from tqdm.notebook import tqdm  # for printing progress bars
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

# Load environment variables
news_api_key = config.NEWS_API_KEY

GPT_MODEL = "gpt-3.5-turbo"


# Helper functions
def json_gpt(input: str):
    completion = client.chat.completions.create(model=GPT_MODEL,
    messages=[
        {"role": "system", "content": "Output only valid JSON"},
        {"role": "user", "content": input},
    ],
    temperature=0.5)

    text = completion.choices[0].message.content
    parsed = json.loads(text)

    return parsed

def embeddings(input: list[str]) -> list[list[str]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=input)
    return [data.embedding for data in response.data]

# User asks a question
USER_QUESTION = "Who won the WNBA championship? And who was the MVP? Tell me a bit about the last game."

QUERIES_INPUT = f"""
You have access to a search API that returns recent news articles.
Generate an array of search queries that are relevant to this question.
Use a variation of related keywords for the queries, trying to be as general as possible.
Include as many queries as you can think of, including and excluding terms.
For example, include queries like ['keyword_1 keyword_2', 'keyword_1', 'keyword_2'].
Be creative. The more queries you include, the more likely you are to find relevant results.

User question: {USER_QUESTION}

Format: {{"queries": ["query_1", "query_2", "query_3"]}}
"""

queries = json_gpt(QUERIES_INPUT)["queries"]

# Let's include the original question as well for good measure
queries.append(USER_QUESTION)

print(queries)

def search_news(
    query: str,
    news_api_key: str = news_api_key,
    num_articles: int = 50,
    from_datetime: str = "2024-10-06",  # the 2024 WNBA finals were played in Oct 2024
    to_datetime: str = "2024-10-31",
) -> dict:
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "apiKey": news_api_key,
            "pageSize": num_articles,
            "sortBy": "relevancy",
            "from": from_datetime,
            "to": to_datetime,
        },
    )

    return response.json()


articles = []

for query in tqdm(queries):
    result = search_news(query)
    if result["status"] == "ok":
        articles = articles + result["articles"]
    else:
        raise Exception(result["message"])

# remove duplicates
articles = list({article["url"]: article for article in articles}.values())

print("Total number of articles:", len(articles))
print("Top 5 articles of query 1:", "\n")

for article in articles[0:5]:
    print("Title:", article["title"])
    print("Description:", article["description"])
    print("Content:", article["content"][0:100] + "...")
    print()

HA_INPUT = f"""
Generate a hypothetical answer to the user's question. This answer will be used to rank search results. 
Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
like NAME did something, or NAME said something at PLACE. 

User question: {USER_QUESTION}

Format: {{"hypotheticalAnswer": "hypothetical answer text"}}
"""

hypothetical_answer = json_gpt(HA_INPUT)["hypotheticalAnswer"]

print(hypothetical_answer)

hypothetical_answer_embedding = embeddings(hypothetical_answer)[0]
article_embeddings = embeddings(
    [
        f"{article['title']} {article['description']} {article['content'][0:100]}"
        for article in articles
    ]
)

# Calculate cosine similarity
cosine_similarities = []
for article_embedding in article_embeddings:
    cosine_similarities.append(dot(hypothetical_answer_embedding, article_embedding))

print(cosine_similarities[0:10])

scored_articles = zip(articles, cosine_similarities)

# Sort articles by cosine similarity
sorted_articles = sorted(scored_articles, key=lambda x: x[1], reverse=True)

# Print top 5 articles
print("Top 5 articles:", "\n")

for article, score in sorted_articles[0:5]:
    print("Title:", article["title"])
    print("Description:", article["description"])
    print("Content:", article["content"][0:100] + "...")
    print("Score:", score)
    print()

formatted_top_results = [
    {
        "title": article["title"],
        "description": article["description"],
        "url": article["url"],
    }
    for article, _score in sorted_articles[0:5]
]

ANSWER_INPUT = f"""
Generate an answer to the user's question based on the given search results. 
TOP_RESULTS: {formatted_top_results}
USER_QUESTION: {USER_QUESTION}

Include as much information as possible in the answer. Reference the relevant search result urls as markdown links.
"""

completion = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[{"role": "user", "content": ANSWER_INPUT}],
    temperature=0.5,
)

print(completion.choices[0].message.content)    
# text = ""
# for chunk in completion:
#     text += chunk.choices[0].delta.content
#     display.clear_output(wait=True)
#     display.display(display.Markdown(text))