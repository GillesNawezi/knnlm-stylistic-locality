import pandas as pd
import numpy as np
import os
import pathlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
tqdm.pandas()

global_path = str(pathlib.Path(__file__).parent.resolve())



#Supportive
supportive =  pd.read_pickle(global_path + "/datasets/prepro/ruddit_final.pklxz",compression="xz")
supportive = supportive.rename(columns={"comment":"text"})
supportive["style"] = np.where(
                        supportive['offensiveness_score'].between(-1, -0.3, inclusive="both"), 
                        'supportive', 
                        np.where(
                                supportive['offensiveness_score'].between(0.3, 1, inclusive="both"), 'offensive', 'neutral_offensive'
                        )
                     )

supportive["source"] = "ruddit"
supportive = supportive[["text","style","source"]]

#Formal
formality = pd.read_pickle(global_path + "/datasets/prepro/formality_gyafc.pklxz",compression="xz")

formality = formality.rename(columns={0:"text"})
formality["source"] = "gyafc"
formality = formality[["text","style","source"]]

#Politness
politeness = pd.read_pickle(global_path + "/datasets/prepro/politeness_wiki_stanford.pklxz",compression="xz")

pol_mapping = {1:"polite",
    	   0: "neutral_polite",
           -1:"impolite"
        }

politeness["style"] = politeness["meta.Binary"].map(pol_mapping)
politeness["source"] = "polite_wiki_talk"

politeness = politeness[["text","style","source"]]


#Toxic
toxicity = pd.read_pickle(global_path + "/datasets/prepro/toxicity_personal_attacks_wiki.pklxz",compression="xz")
tox_mapping = {True:"toxic",False:"non-toxic"}
toxicity["style"] = toxicity["attack"].map(tox_mapping)
toxicity["source"] = "toxic_wiki_talk"

toxicity = toxicity.rename(columns={"comment":"text"})
toxicity = toxicity[["text","style","source"]]


#Civil Comments Toxicity
toxicity_cv = pd.read_pickle(global_path + "/datasets/prepro/civil_comments.pklxz",compression="xz")
toxicity_cv = toxicity_cv.rename(columns={"comment_text":"text"})
toxicity_cv["style"] = np.where(toxicity_cv["toxicity"]>=0.5,"toxic","non-toxic")
toxicity_cv["source"] = "civil_comments"
toxicity_cv = toxicity_cv[["text","style","source"]]


style_df = pd.concat([supportive,toxicity,toxicity_cv,politeness,formality], ignore_index=True)
style_df["text"] = style_df["text"].str.strip()

#Filter deleted/removed Messages
filter = ["[deleted]","[removed]",]
style_df = style_df[~style_df["text"].isin(filter)]

#Filter unwanted Styles
filter_style = ["non_toxic","non-toxic","neutral_polite","neutral_offensive"]
style_df = style_df[~style_df["style"].isin(filter_style)]

#Get Style source
positive_styles = ["supportive","polite","formal"]
style_df["category"] = np.where(style_df["style"].isin(positive_styles),"pos","neg")

style_df = style_df.replace(r'\n',' ', regex=True) 
print(f"Before dropna: {len(style_df)}")
style_df = style_df.dropna().reset_index(drop=True)
print(f"After dropna:{len(style_df)}")

style_df["text"] = style_df["text"].str.replace('([.,!?()])', r' \1 ').str.replace('\s{2,}', ' ')
style_df["text"] = style_df["text"].str.strip('\"').str.replace('\"',' \" ').str.replace("\'", " \' ")

#import string
#style_df["text"] =  style_df['text'].str.replace('[{}]'.format(string.punctuation), '')

#Downsampling

def to_txt(writePath, df):
        with open(writePath, 'a') as f:
                dfAsString = df.to_string(header=False, index=False)
                f.write(dfAsString)


def create_input_files(df, dir_name):
        X_train, X_test, y_train, y_test = train_test_split(df[["text","source","category"]], df["style"], random_state=0, test_size=0.15, stratify=df["style"])
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.15, stratify=y_train)

        if not os.path.exists(dir_name):
                os.mkdir(dir_name)

        src_train = X_train["source"]
        src_test = X_test["source"]
        src_valid = X_valid["source"]

        cat_train = X_train["category"]
        cat_test = X_test["category"]
        cat_valid = X_valid["category"]

        X_train = X_train.drop(columns=["source","category"])
        X_test = X_test.drop(columns=["source","category"])
        X_valid = X_valid.drop(columns=["source","category"])
        
        import csv 
        src_train.to_csv(dir_name + "train.txt.source", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        src_test.to_csv(dir_name + "test.txt.source", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        src_valid.to_csv(dir_name + "valid.txt.source", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        cat_train.to_csv(dir_name + "train.txt.category", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        cat_test.to_csv(dir_name + "test.txt.category", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        cat_valid.to_csv(dir_name + "valid.txt.category", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        
        X_train.to_csv(dir_name + "train.txt", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        X_test.to_csv(dir_name + "test.txt", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        X_valid.to_csv(dir_name + "valid.txt", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        y_train.to_csv(dir_name + "train.txt.style", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        y_test.to_csv(dir_name + "test.txt.style", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        y_valid.to_csv(dir_name + "valid.txt.style", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        pd.concat([src_test,src_train], ignore_index=True).to_csv(dir_name + "testtrain.txt.source", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        pd.concat([src_valid,src_train], ignore_index=True).to_csv(dir_name + "validtrain.txt.source", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        pd.concat([cat_test,cat_train], ignore_index=True).to_csv(dir_name + "testtrain.txt.category", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        pd.concat([cat_valid,cat_train], ignore_index=True).to_csv(dir_name + "validtrain.txt.category", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        pd.concat([X_test,X_train], ignore_index=True).to_csv(dir_name + "testtrain.txt", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        pd.concat([X_valid,X_train], ignore_index=True).to_csv(dir_name + "validtrain.txt", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        pd.concat([y_test,y_train], ignore_index=True).to_csv(dir_name + "testtrain.txt.style", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)
        pd.concat([y_valid,y_train], ignore_index=True).to_csv(dir_name + "validtrain.txt.style", header=None, index=None, sep=' ', mode='w', quoting=csv.QUOTE_ALL)

        print("Done")

        print(len(X_train))
        print(sum(1 for line in open(dir_name + "train.txt")))
        print(sum(1 for line in open(dir_name + "train.txt.style")))
        print(sum(1 for line in open(dir_name + "train.txt.source")))
        print(sum(1 for line in open(dir_name + "train.txt.category")))
        print("\n")
        print(len(X_valid))
        print(sum(1 for line in open(dir_name + "valid.txt")))
        print(sum(1 for line in open(dir_name + "valid.txt.style")))
        print(sum(1 for line in open(dir_name + "valid.txt.source")))
        print(sum(1 for line in open(dir_name + "valid.txt.category")))
        print("\n")




dir_name = global_path + "/output/style_category_dataset/"

create_input_files(style_df, dir_name)
print(len(style_df))

files = os.listdir(dir_name)

for file in tqdm(files):
        file = dir_name + file
        tqdm.write(file)
        with open(file, "r") as f:
                lines = f.readlines()
        os.remove(file)
        with open(file, "w") as f:
                for line in lines:
                        line = line.strip().strip('"').replace('""','"')
                        f.write(line)
                        f.write("\n")
                
print("DONE")
"""
style_df["no_of_words"] = style_df["text"].str.split().str.len()
no_of_tokens = style_df["no_of_words"].sum() 
print(f"No of Tokens {no_of_tokens}")
"""

