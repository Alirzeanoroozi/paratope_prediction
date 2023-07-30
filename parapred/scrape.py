import pandas as pd
from tqdm import tqdm
import urllib.request
import os

if __name__ == "__main__":
    dir_list = os.listdir("data/pdb")
    print(dir_list)

    df = pd.read_csv("data/dataset.csv")
    for pdb_name in tqdm(df['pdb']):
        if pdb_name + ".pdb" in dir_list:
            continue
        else:
            urllib.request.urlretrieve("https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/" + pdb_name + "/?scheme=chothia", "data/pdb/" + pdb_name + ".pdb")