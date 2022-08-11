"""
Load Ruddit Dataset and load the actual comments by using the provided by the csv file
"""

import pandas as pd
import pathlib
import praw
from tqdm import tqdm
tqdm.pandas()

global_path = str(pathlib.Path(__file__).parent.resolve())

client_id = "i3BnN3dhO6TxgVdk1o4g1w"
client_secret = "uMpmWAjxboAywVIvX9Pi9e0mMNvnvQ"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent ="ruddit_comment_loader"
)


df = pd.read_csv(global_path + "/datasets/ruddit/Ruddit.csv")

def load_comment_for_post(comment_id):
    try:
        comment = reddit.comment(comment_id)
        return {"comment":comment.body, "author":comment.author}
    except Exception as e:     
        return {"comment":e, "author":e}


df["comment_info"] = df["comment_id"].progress_apply(load_comment_for_post)
df = df.join(df["comment_info"].apply(pd.Series))
df = df.drop(columns=["comment_info"])

df.to_pickle(global_path + "/datasets/prepro/ruddit_final.pklxz",compression="xz")