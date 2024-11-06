import openai
import config
import praw
import pandas as pd
from typing import List, Dict, Generator, Optional



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

NUM_SUBREDDITS = len(subreddits)
TARGET_COMMENTS_PER_SUBREDDIT = 50
MAX_COMMENTS_PER_SUBMISSION = 10
MAX_COMMENT_LENGTH = 2000

collect_comments(
    filename=filename,
    target_comments_per_subreddit=TARGET_COMMENTS_PER_SUBREDDIT,
    max_comments_per_submission=MAX_COMMENTS_PER_SUBMISSION,
    max_comment_length=MAX_COMMENT_LENGTH,
    reddit=reddit,
)