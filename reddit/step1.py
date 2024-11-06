import openai
import config
import praw


model = "gpt-3.5-turbo"

openai.api_key = config.OPENAI_API_KEY


reddit = praw.Reddit(
    client_id=config.REDDIT_CLIENT_ID,
    client_secret=config.REDDIT_CLIENT_SECRET,
    user_agent=f"script:test:0.0.1 (by u/WaferNew2080)",
)
print(reddit.read_only)

for submission in reddit.subreddit("test").hot(limit=10):
    print(submission.title)