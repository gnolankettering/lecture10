import openai
import config
import praw
import pandas as pd
from typing import List, Dict, Generator, Optional
import re

from tenacity import (
    retry,
    wait_random_exponential,
    retry_if_exception_type,
    stop_after_attempt,
)

model = "gpt-3.5-turbo"

openai.api_key = config.OPENAI_API_KEY


reddit = praw.Reddit(
    client_id=config.REDDIT_CLIENT_ID,
    client_secret=config.REDDIT_CLIENT_SECRET,
    user_agent=f"script:test:0.0.1 (by u/WaferNew2080)",
)
# print(reddit.read_only)

# for submission in reddit.subreddit("test").hot(limit=10):
#     print(submission.title)

DF_COLUMNS = ["subreddit", "submission_id", "score", "comment_body"]
filename, subreddits = (
    "epl_top_8.csv",
    [
        "reddevils",
        "LiverpoolFC",
        "chelseafc",
        "Gunners",
        "coys",
        "MCFC",
        "Everton",
        "NUFC",
    ],
)

# Utility functions for fetching comments from submissions
def comment_generator(submission) -> Generator:
    # Do not bother expanding MoreComments (follow-links)
    for comment in submission.comments.list():
        if (
            hasattr(comment, "body")
            and comment.body != "[deleted]"
            and comment.body != "[removed]"
        ):
            yield (comment)

def collect_comments(
    filename: str,
    target_comments_per_subreddit: int,
    max_comments_per_submission: int,
    max_comment_length: int,
    reddit: praw.Reddit,
) -> pd.DataFrame:
    """
    Collect comments from the top submissions in each subreddit.

    Cache results at cache_filename.

    Return a dataframe with columns: subreddit, submission_id, score, comment_body
    """
    try:
        df = pd.read_csv(filename, index_col="id")
        assert df.columns.tolist() == DF_COLUMNS
    except FileNotFoundError:
        df = pd.DataFrame(columns=DF_COLUMNS)

    # dict like {comment_id -> {column -> value}}
    records = df.to_dict(orient="index")

    for subreddit_index, subreddit_name in enumerate(subreddits):
        print(f"Processing Subreddit: {subreddit_name}")

        processed_comments_for_subreddit = len(df[df["subreddit"] == subreddit_name])

        if processed_comments_for_subreddit >= target_comments_per_subreddit:
            print(
                f"Enough comments fetched for {subreddit_name}, continuing to next subreddit."
            )
            continue

        # `top`` is a generator, grab submissions until we break (within this loop).
        for submission in reddit.subreddit(subreddit_name).top(time_filter="month"):
            if processed_comments_for_subreddit >= target_comments_per_subreddit:
                break

            # The number of comments that we already have for this subreddit
            processed_comments_for_submission = len(
                df[df["submission_id"] == submission.id]
            )

            for comment in comment_generator(submission):
                if (
                    processed_comments_for_submission >= max_comments_per_submission
                    or processed_comments_for_subreddit >= target_comments_per_subreddit
                ):
                    break

                if comment.id in records:
                    print(
                        f"Skipping comment {subreddit_name}-{submission.id}-{comment.id} because we already have it"
                    )
                    continue

                body = comment.body[:max_comment_length].strip()
                records[comment.id] = {
                    "subreddit": subreddit_name,
                    "submission_id": submission.id,
                    "comment_body": body,
                }

                processed_comments_for_subreddit += 1
                processed_comments_for_submission += 1

            # Once per post write to disk.
            print(f"CSV rewritten with {len(records)} rows.\n")
            df = pd.DataFrame.from_dict(records, orient="index", columns=DF_COLUMNS)
            df.to_csv(filename, index_label="id")

    print("Completed.")
    return df

# NUM_SUBREDDITS = len(subreddits)
# TARGET_COMMENTS_PER_SUBREDDIT = 50
# MAX_COMMENTS_PER_SUBMISSION = 10
# MAX_COMMENT_LENGTH = 2000

# collect_comments(
#     filename=filename,
#     target_comments_per_subreddit=TARGET_COMMENTS_PER_SUBREDDIT,
#     max_comments_per_submission=MAX_COMMENTS_PER_SUBMISSION,
#     max_comment_length=MAX_COMMENT_LENGTH,
#     reddit=reddit,
# )

MAX_ATTEMPTS = 3


def generate_prompt_messages(s: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": """
The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

The traffic is quite annoying.
""".strip(),
        },
        {"role": "assistant", "content": "-0.75"},
        {
            "role": "user",
            "content": """
The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

The library is downtown.
""".strip(),
        },
        {"role": "assistant", "content": "0.0"},
        {
            "role": "user",
            "content": """
The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

Even though it's humid, I really love the summertime. Everything is so green and the sun is out all the time.
""".strip(),
        },
        {"role": "assistant", "content": "0.8"},
        {
            "role": "user",
            "content": f"""
The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

{s}
""".strip(),
        },
    ]

class UnscorableCommentError(Exception):
    pass

@retry(
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(UnscorableCommentError)
    | retry_if_exception_type(openai.APIConnectionError)
    | retry_if_exception_type(openai.APIError)
    | retry_if_exception_type(openai.RateLimitError),
    reraise=True,  # Reraise the last exception
)
def score_sentiment(s: str, model: str) -> float:
    messages = generate_prompt_messages(s)
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    score_response = response.choices[0].message.content.strip()
    # This will raise an Attribute Error if the regular expression doesn't match
    try:
        return float(re.search(r"([-+]?\d*\.?\d+)", score_response).group(1))
    except AttributeError:
        raise UnscorableCommentError(f"Could not score comment: {s}")
    

def score_sentiments(filename: str, model: str) -> pd.DataFrame:
    """
    Score sentiments contained in comments in filename.
    """
    df = pd.read_csv(filename, index_col="id")
    assert df.columns.tolist() == DF_COLUMNS

    records = df.to_dict(orient="index")

    for index, item in enumerate(records.items()):
        comment_id, comment = item

        if not pd.isna(comment["score"]):
            print(f"{comment_id} was already scored. Skipping.")
            continue

        body = comment["comment_body"]
        try:
            score = score_sentiment(body, model=model)
        except UnscorableCommentError:
            # The score_sentiment method will retry 3 times before letting this error pass through.
            # If it does, we will consider this comment un-processable and skip it.
            # For other errors, such as APIConnectionError, we will fail completely and let the user know.
            continue
        print(
            f"""
            {comment_id} - ({index + 1} of {len(records)} Comments)
            Body: {body[:80]}
            Score: {score}""".strip()
        )

        records[comment_id]["score"] = score
        df = pd.DataFrame.from_dict(records, orient="index", columns=DF_COLUMNS)
        df.to_csv(filename, index_label="id")

    print("Scoring completed.")
    return df

df = score_sentiments(filename=filename, model=model)
