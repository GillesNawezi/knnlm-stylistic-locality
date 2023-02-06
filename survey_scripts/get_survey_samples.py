import pandas as pd
import numpy as np
import os
import pathlib
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split
from itertools import groupby 
from string import punctuation
tqdm.pandas()


global_path = str(pathlib.Path(__file__).parent.parent.resolve())

#dir_name = global_path + "/output/offensive_dataset/"

dir_name = "/mnt/g/projects/knnlm-stylistic-locality/style_data_prepro/output/style_source_neutral/"




with open(dir_name + "test.txt", "r") as f:
    X_test = f.readlines()

with open(dir_name + "test.txt.style", "r") as f:
    y_test = f.readlines()

with open(dir_name + "test.txt.source", "r") as f:
    y_test_source = f.readlines()


df_test = pd.DataFrame(
    { 'text': X_test,
      'style': y_test,
      'source': y_test_source 
      })

def sampling_k_elements(group, k=5):
    if len(group) < k:
        return group
    return group.sample(k)

df_test != df_test.query("style!='offensive'")

i=1
while (i <= 7):
    reduced_test = df_test.groupby('style').apply(sampling_k_elements, k=5)

    print(len(df_test))
    rem_ix = reduced_test.index.to_frame()[1].tolist()
    df_test = df_test.drop(index=rem_ix)
    print(len(df_test))

    reduced_test = reduced_test.reset_index(drop=True)
    out_dir_name = "/mnt/g/projects/knnlm-stylistic-locality/style_data_prepro/output/survey/"
    out_file = out_dir_name + f"survey_samples_{str(i)}.txt"

    try:
        os.remove(out_file)
    except:
        pass

    punc = set(punctuation) - set('.')
    samples = reduced_test["text"].tolist()

    with open(out_file, "w+") as f: 
        for sample in samples:


            newtext = []
            for k, g in groupby(sample):
                if k in punc:
                    newtext.append(k)
                else:
                    newtext.extend(g)

            sample = ''.join(newtext)

            sample = sample.split(" ")


            new_len = int(np.ceil(len(sample) / 2))

            sample = " ".join(sample[:new_len])

            f.write(sample)
            f.write("\n")
    i+=1




