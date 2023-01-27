import pandas as pd
import numpy as np
import os
import pathlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
tqdm.pandas()

global_path = str(pathlib.Path(__file__).parent.resolve())


dir_name = global_path + "/output/category_source/"


files = os.listdir(dir_name)

positive_styles = ["supportive","polite","formal"]

for file in files:
    print(file)
    if "style" in file:
        with open(dir_name + file, "r") as f:
            lines = f.readlines()
        os.remove(dir_name + file)

        with open(dir_name + file, "w") as f:
            for line in lines:
                line = line.replace("\n","")
                print(line)
                if line in positive_styles: 
                    f.write("pos")
                else: 
                    f.write("neg")
                f.write("\n")
        

