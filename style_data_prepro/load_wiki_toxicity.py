import pandas as pd
import urllib
import pathlib


global_path = str(pathlib.Path(__file__).parent.resolve())


# download annotated comments and annotations

ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634' 
ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637' 


def download_file(url, fname):
    urllib.request.urlretrieve(url, fname)

                
download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')


comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)


annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

comments['attack'] = labels

comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

df = comments.copy()

df.to_pickle(global_path+ "/datasets/prepro/" + "toxicity_personal_attacks_wiki.pklxz",compression="xz")

