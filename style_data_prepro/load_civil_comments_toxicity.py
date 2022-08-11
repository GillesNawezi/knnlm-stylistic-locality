"""
Load Jigsaw Comments
"""

import pandas as pd
import pathlib
import praw
from tqdm import tqdm
tqdm.pandas()

global_path = str(pathlib.Path(__file__).parent.resolve())

data_path =  global_path + "/datasets/civil_comments/all_data.csv"


df = pd.read_csv(data_path)
print("test")
df.to_pickle(global_path + "/datasets/prepro/civil_comments.pklxz",compression="xz")