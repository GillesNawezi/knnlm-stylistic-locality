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
supportive = supportive[["text","style"]]

#Formal
formality = pd.read_pickle(global_path + "/datasets/prepro/formality_gyafc.pklxz",compression="xz")
formality = formality.rename(columns={0:"text"})
formality = formality[["text","style"]]

#Politness
politeness = pd.read_pickle(global_path + "/datasets/prepro/politeness_wiki_stanford.pklxz",compression="xz")

pol_mapping = {1:"polite",
    	   0: "neutral_polite",
           -1:"impolite"
        }

politeness["style"] = politeness["meta.Binary"].map(pol_mapping)
politeness = politeness[["text","style"]]

#Toxic
toxicity = pd.read_pickle(global_path + "/datasets/prepro/toxicity_personal_attacks_wiki.pklxz",compression="xz")
tox_mapping = {True:"toxic",False:"non-toxic"}
toxicity["style"] = toxicity["attack"].map(tox_mapping)

toxicity = toxicity.rename(columns={"comment":"text"})
toxicity = toxicity[["text","style"]]


#Civil Comments Toxicity
toxicity_cv = pd.read_pickle(global_path + "/datasets/prepro/civil_comments.pklxz",compression="xz")
toxicity_cv = toxicity_cv.rename(columns={"comment_text":"text"})
toxicity_cv["style"] = np.where(toxicity_cv["toxicity"]>=0.5,"toxic","non-toxic")
toxicity_cv = toxicity_cv[["text","style"]]


style_df = pd.concat([supportive,toxicity,toxicity_cv,politeness,formality], ignore_index=True)
style_df["text"] = style_df["text"].str.strip()

#Filter deleted/removed Messages
filter = ["[deleted]","[removed]",]
style_df = style_df[~style_df["text"].isin(filter)]

#Filter unwanted Styles
filter_style = ["non_toxic","non-toxic","neutral_polite","neutral_offensive"]
style_df = style_df[~style_df["style"].isin(filter_style)]

#Get Style Category
positive_styles = ["supportive","polite","formal"]
style_df["category"] = np.where(style_df["style"].isin(positive_styles),"pos","neg")

style_df = style_df.replace(r'\n',' ', regex=True) 
print(f"Before dropna: {len(style_df)}")
style_df = style_df.dropna().reset_index(drop=True)
print(f"After dropna:{len(style_df)}")
x=y
#Downsampling
"""
no_of_samples = 3000
styles = style_df["style"].unique().tolist()


sample_df = style_df.groupby("style").sample(n=3000, random_state=1, replace=True).reset_index(drop=True)
"""
def to_txt(writePath, df):
        with open(writePath, 'a') as f:
                dfAsString = df.to_string(header=False, index=False)
                f.write(dfAsString)


def create_input_files(df, dir_name):
        X_train, X_test, y_train, y_test = train_test_split(df[["text","category"]], df["style"], random_state=0, test_size=0.15, stratify=df["style"])
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.15, stratify=y_train)

        if not os.path.exists(dir_name):
                os.mkdir(dir_name)

        cat_train = X_train["category"]
        cat_test = X_test["category"]
        cat_valid = X_valid["category"]

        X_train = X_train.drop(columns=["category"])
        X_test = X_test.drop(columns=["category"])
        X_valid = X_valid.drop(columns=["category"])

        cat_train.to_csv(dir_name + "train.txt.category", header=None, index=None, sep=' ', mode='w')
        cat_test.to_csv(dir_name + "test.txt.category", header=None, index=None, sep=' ', mode='w')
        cat_valid.to_csv(dir_name + "valid.txt.category", header=None, index=None, sep=' ', mode='w')
        
        X_train.to_csv(dir_name + "train.txt", header=None, index=None, sep=' ', mode='w')
        X_test.to_csv(dir_name + "test.txt", header=None, index=None, sep=' ', mode='w')
        X_valid.to_csv(dir_name + "valid.txt", header=None, index=None, sep=' ', mode='w')

        y_train.to_csv(dir_name + "train.txt.style", header=None, index=None, sep=' ', mode='w')
        y_test.to_csv(dir_name + "test.txt.style", header=None, index=None, sep=' ', mode='w')
        y_valid.to_csv(dir_name + "valid.txt.style", header=None, index=None, sep=' ', mode='w')

        pd.concat([cat_test,cat_train], ignore_index=True).to_csv(dir_name + "testtrain.txt.category", header=None, index=None, sep=' ', mode='w')
        pd.concat([cat_valid,cat_train], ignore_index=True).to_csv(dir_name + "validtrain.txt.category", header=None, index=None, sep=' ', mode='w')

        pd.concat([X_test,X_train], ignore_index=True).to_csv(dir_name + "testtrain.txt", header=None, index=None, sep=' ', mode='w')
        pd.concat([X_valid,X_train], ignore_index=True).to_csv(dir_name + "validtrain.txt", header=None, index=None, sep=' ', mode='w')

        pd.concat([y_test,y_train], ignore_index=True).to_csv(dir_name + "testtrain.txt.style", header=None, index=None, sep=' ', mode='w')
        pd.concat([y_valid,y_train], ignore_index=True).to_csv(dir_name + "validtrain.txt.style", header=None, index=None, sep=' ', mode='w')

        print("Done")

        print(len(X_train))
        print(sum(1 for line in open(dir_name + "train.txt")))
        print(sum(1 for line in open(dir_name + "train.txt.style")))
        print(sum(1 for line in open(dir_name + "train.txt.category")))
        print("\n")
        print(len(X_valid))
        print(sum(1 for line in open(dir_name + "valid.txt")))
        print(sum(1 for line in open(dir_name + "valid.txt.style")))
        print(sum(1 for line in open(dir_name + "valid.txt.category")))
        print("\n")
        print(len(pd.concat([X_valid,X_train])))
        print(len(pd.concat([cat_valid,cat_train])))



dir_name = global_path + "/output/style_category_dataset/"

create_input_files(style_df, dir_name)
print(len(style_df))
"""
style_df["no_of_words"] = style_df["text"].str.split().str.len()
no_of_tokens = style_df["no_of_words"].sum() 
print(f"No of Tokens {no_of_tokens}")
"""

