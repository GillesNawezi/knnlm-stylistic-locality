import os
import pandas as pd
import pathlib
import praw
from tqdm import tqdm
tqdm.pandas()

global_path = str(pathlib.Path(__file__).parent.resolve())

dir1 = global_path + "/datasets/GYAFC_Corpus/Entertainment_Music/"
dir2 = global_path + "/datasets/GYAFC_Corpus/Family_Relationships/"
topics = {"Entertainment_Music":dir1, "Family_Relationships":dir2}

 #Formal
test = os.listdir(dir1 + "test") 
train = os.listdir(dir1 + "train") 
tune = os.listdir(dir1 + "tune") 
splits = {"train","test","tune"}

df = pd.DataFrame()

for topic, dir in topics.items():
    print(topic)
    for split in splits:
        files = os.listdir(dir1 + split) 
        for file in files:
            print(file)
            file_df = pd.read_csv(dir + split + "/" + file, delimiter = "\t",header=None)
            if "informal" in file: 
                file_df["style"] = "informal"
            elif "formal" in file:
                file_df["style"] = "formal"
            file_df["topic"] = topic
            file_df["src_file"] = file
            file_df["split"] = split
            df= pd.concat([df,file_df],ignore_index=True)


df.to_pickle(global_path+ "/datasets/prepro/" + "formality_gyafc.pklxz",compression="xz")

test = pd.read_pickle(global_path+ "/datasets/prepro/" + "formality_gyafc.pklxz",compression="xz")
#INformal